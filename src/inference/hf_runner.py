"""Transformers runner, used for the five models vLLM does not cover here.

Two details are load-bearing and were both learned the hard way.

The image goes *inside* the chat message, and `images=` is not also passed.
Passing both either raises on a duplicate argument or, through the older
processor path, leaves a prompt with zero image tokens -- the model then answers
from the text alone and nothing in the output says so.

The model is pinned to the single visible GPU. `device_map="auto"` quietly
offloads layers to CPU when a co-tenant has taken the memory, which cost ~17 s
an item against ~0.5 s on an uncontended GPU. Pinning either keeps the model on
the GPU or fails loudly with an OOM.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.inference.base import VLMRunner
from src.utils import get_logger

log = get_logger(__name__)

_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _model_classes() -> list:
    """Auto classes to try, most specific first."""
    import transformers
    names = ["AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"]
    classes = [getattr(transformers, n) for n in names if hasattr(transformers, n)]
    if not classes:
        raise ImportError("transformers exposes no usable vision-language auto class")
    return classes


class HFRunner(VLMRunner):
    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import AutoProcessor

        dtype = (torch.bfloat16 if str(self.model_cfg.get("dtype", "bf16")).startswith("bf")
                 else torch.float16)
        device_map = {"": 0} if torch.cuda.is_available() else None
        log.info(f"loading {self.model_id} via transformers (dtype={dtype})")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, token=_HF_TOKEN)

        last_error: Exception | None = None
        for cls in _model_classes():
            try:
                self.model = cls.from_pretrained(
                    self.model_id, torch_dtype=dtype, device_map=device_map,
                    trust_remote_code=True, token=_HF_TOKEN)
                self.model.eval()
                self._loaded = True
                log.info(f"loaded {self.model_id} with {cls.__name__}")
                return
            except Exception as e:
                last_error = e
                log.warning(f"{cls.__name__} could not load {self.model_id}: {e}")
        raise RuntimeError(f"failed to load {self.model_id}: {last_error}")

    def _encode(self, image: Image.Image, prompt: str, system: str):
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]})
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt")
        except Exception as e:
            # Older processors want a bare {"type": "image"} placeholder and the
            # image passed separately.
            log.debug(f"image-in-content template failed, falling back: {e}")
            placeholder = []
            if system:
                placeholder.append({"role": "system", "content": system})
            placeholder.append({"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt}]})
            text = self.processor.apply_chat_template(
                placeholder, add_generation_prompt=True, tokenize=False)
            inputs = self.processor(text=text, images=image, return_tensors="pt")
        return inputs.to(self.model.device)

    def generate(self, image_path: str | Path, prompt: str, system: str,
                 max_tokens: int | None = None) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("call load() first")

        started = time.perf_counter()
        image = Image.open(image_path).convert("RGB")
        try:
            inputs = self._encode(image, prompt, system)
        except Exception as e:
            log.error(f"processor failed on {image_path}: {e}")
            return {"text": "", "gen_seconds": time.perf_counter() - started}

        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or self.max_tokens,
                    do_sample=False, temperature=None, top_p=None)
            n_prompt = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            text = self.processor.batch_decode(
                output_ids[:, n_prompt:], skip_special_tokens=True)[0]
        except Exception as e:
            log.error(f"generate failed on {image_path}: {e}")
            text = ""

        return {"text": text, "gen_seconds": time.perf_counter() - started}

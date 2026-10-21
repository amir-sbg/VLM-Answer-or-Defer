"""vLLM runner, used for the three models whose architectures it supports.

Decoding is greedy (temperature 0), so batching changes throughput and nothing
else: the same item gets the same answer whatever it is batched with.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# FlashInfer's sampler JIT-compiles a CUDA kernel when the engine starts, which
# needs nvcc and a CUDA_HOME that compute nodes shipping only the runtime do not
# have. We decode greedily, so the native Torch sampler gives identical output.
# This has to be set before the engine is built; spawned workers inherit it.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from src.inference.base import VLMRunner, encode_image_b64
from src.utils import get_logger

log = get_logger(__name__)

_DTYPE = {"bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16",
          "float16": "float16", "fp32": "float32", "float32": "float32", "auto": "auto"}


class VLLMRunner(VLMRunner):
    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.llm = None
        self.sampling_params = None

    def load(self) -> None:
        from vllm import LLM, SamplingParams

        dtype = _DTYPE.get(str(self.model_cfg.get("dtype", "bf16")).lower(), "bfloat16")
        log.info(f"loading {self.model_id} via vLLM (dtype={dtype})")

        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            dtype=dtype,
            max_model_len=self.model_cfg.get("max_model_len", 8192),
            gpu_memory_utilization=self.model_cfg.get("gpu_memory_utilization", 0.85),
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
        )
        if os.environ.get("HF_TOKEN"):
            kwargs["hf_token"] = os.environ["HF_TOKEN"]

        self.llm = LLM(**kwargs)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=self.max_tokens)
        self._loaded = True

    def _messages(self, image_path: str, prompt: str, system: str) -> list[dict]:
        b64 = encode_image_b64(image_path)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]})
        return messages

    def generate(self, image_path: str | Path, prompt: str, system: str,
                 max_tokens: int | None = None) -> dict[str, Any]:
        return self.generate_batch([(image_path, prompt, system)], max_tokens)[0]

    def generate_batch(self, items: list[tuple[str | Path, str, str]],
                       max_tokens: int | None = None) -> list[dict[str, Any]]:
        if not self._loaded:
            raise RuntimeError("call load() first")

        from vllm import SamplingParams
        params = (SamplingParams(temperature=0.0, max_tokens=max_tokens)
                  if max_tokens is not None else self.sampling_params)

        messages = [self._messages(str(image), prompt, system) for image, prompt, system in items]
        started = time.perf_counter()
        try:
            outputs = self.llm.chat(messages=messages, sampling_params=params)
        except Exception as e:
            log.error(f"vLLM generate failed for a batch of {len(items)}: {e}")
            return [{"text": "", "gen_seconds": 0.0} for _ in items]

        elapsed = (time.perf_counter() - started) / len(items)
        return [{"text": out.outputs[0].text if out.outputs else "", "gen_seconds": elapsed}
                for out in outputs]

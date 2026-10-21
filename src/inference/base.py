"""Runner interface. One instance per model; `generate` is called per item."""
from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


def encode_image_b64(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class VLMRunner(ABC):
    def __init__(self, model_cfg: dict):
        self.model_cfg = model_cfg
        self.model_id = model_cfg["id"]
        self.max_tokens = model_cfg.get("max_tokens", 512)
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load weights. Called once, before any generate()."""

    @abstractmethod
    def generate(self, image_path: str | Path, prompt: str, system: str,
                 max_tokens: int | None = None) -> dict[str, Any]:
        """Answer one (image, prompt). Returns {"text", "gen_seconds"}."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.model_id})"


def make_runner(model_cfg: dict, defaults: dict | None = None) -> VLMRunner:
    cfg = {**(defaults or {}), **model_cfg}
    backend = cfg.get("backend", "hf")
    if backend == "vllm":
        from src.inference.vllm_runner import VLLMRunner
        return VLLMRunner(cfg)
    if backend == "hf":
        from src.inference.hf_runner import HFRunner
        return HFRunner(cfg)
    raise ValueError(f"unknown backend: {backend}")

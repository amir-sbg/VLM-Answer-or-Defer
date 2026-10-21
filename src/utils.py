"""Small shared helpers: project paths, JSONL and CSV I/O, seeds, logging.

Everything here is deliberately dependency-light so that the analysis modules
can be imported without torch or transformers installed.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    """Resolve a path that may be given relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ── configuration ─────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> dict:
    import yaml
    with open(project_path(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_study_config(path: str | Path = "config/study.yaml") -> dict:
    """Load the study constants (thresholds, budgets, bins, bootstrap settings)."""
    return load_yaml(path)


def model_labels(path: str | Path = "config/models.yaml") -> dict[str, str]:
    """Map Hugging Face model id -> the short label used in the paper's tables.

    Insertion order is the order models appear in Table 2, and dicts preserve it,
    so callers can use this to order rows without a second list.
    """
    cfg = load_yaml(path)
    return {m["id"]: m["label"] for m in cfg["models"]}


# ── JSONL ─────────────────────────────────────────────────────────────────────

def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file, skipping blank and malformed lines.

    A run killed mid-write leaves a truncated final line; skipping it is what
    makes a resumed run pick up cleanly instead of dying on the tail.
    """
    path = Path(path)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def load_done_keys(path: str | Path, key_fields: tuple[str, ...]) -> set[tuple]:
    """Key tuples already present in an output file, for resuming a run."""
    return {tuple(rec.get(f) for f in key_fields) for rec in iter_jsonl(path)}


# ── CSV ───────────────────────────────────────────────────────────────────────

def write_csv(frame, path: str | Path, **kwargs) -> Path:
    """Write a DataFrame atomically and return the path.

    Atomic because the analysis targets are re-run often and a half-written CSV
    read by the next stage is a silent wrong number, not a crash.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False, **kwargs)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


# ── randomness ────────────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def shuffle_options(options: list[str], item_seed: int) -> tuple[list[str], list[int]]:
    """Shuffle multiple-choice options for one item, reproducibly.

    Returns the shuffled options and the original indices, so grading can map a
    chosen letter back to the label it stood for.
    """
    rng = make_rng(item_seed)
    indices = list(range(len(options)))
    rng.shuffle(indices)
    return [options[i] for i in indices], indices


# ── logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

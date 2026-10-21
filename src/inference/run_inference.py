"""Run one model over the exam and append its transcripts to a JSONL file.

Runs are resumable and shardable, which together are what make a 4,000-slice
exam across eight models survive a cluster. Resume skips any
(model, image_id, prompt_id) already in the output file. Sharding splits the
prompts by a stable hash of the prompt identity, so each GPU writes its own
file and no two processes ever append to the same one.

The hash is md5 of "image_id|prompt_id", not Python's hash(), which is salted
per process: with hash() the four workers would disagree about who owns what
and the run would both duplicate and miss items. Hashing the prompt identity
rather than a list index also keeps a prompt in the same shard when the set of
formats changes, so a later run resumes against the right file.

Usage:
    python -m src.inference.run_inference \
        --model google/gemma-4-12B-it \
        --manifest data/exam_set.csv \
        --output-dir raw_responses
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

from src.inference.base import make_runner
from src.prompts.render import render_all
from src.utils import append_jsonl, get_logger, load_done_keys, load_yaml, setup_logging

log = get_logger(__name__)

RESUME_KEY = ("model", "image_id", "prompt_id")


def select_models(models_cfg: dict, model_id: str | None, array_index: int | None) -> list[dict]:
    models = models_cfg.get("models", [])
    if model_id:
        chosen = [m for m in models if m["id"] == model_id]
        if not chosen:
            sys.exit(f"model not in the registry: {model_id}")
        return chosen
    if array_index is not None:
        if not 0 <= array_index < len(models):
            sys.exit(f"array index {array_index} out of range ({len(models)} models)")
        return [models[array_index]]
    return models


def shard_of(row: dict, n_shards: int) -> int:
    key = f"{row['image_id']}|{row['prompt_id']}".encode()
    return int(hashlib.md5(key).hexdigest(), 16) % n_shards


def attach_image_paths(rows: list[dict], manifest: pd.DataFrame) -> list[dict]:
    by_id = dict(zip(manifest["image_id"], manifest["path"]))
    for row in rows:
        row["image_path"] = by_id.get(row["image_id"], "")
    return [r for r in rows if r["image_path"]]


def transcript(model_id: str, row: dict, result: dict) -> dict:
    from datetime import datetime, timezone
    return {
        "model": model_id,
        "image_id": row["image_id"],
        "prompt_id": row["prompt_id"],
        "task": row["task"],
        "phrasing": row["phrasing"],
        "format": row["format"],
        "mc_options": row.get("mc_options", []),
        "mc_option_order": row.get("mc_option_order", []),
        "mc_correct_letter": row.get("mc_correct_letter", ""),
        "image_path": row["image_path"],
        "response": result.get("text", ""),
        "gen_seconds": round(result.get("gen_seconds", 0.0), 3),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_model(model_cfg: dict, defaults: dict, rows: list[dict], output_dir: Path,
              batch_size: int, suffix: str, dry_run: bool) -> None:
    model_id = model_cfg["id"]
    out_path = output_dir / f"{model_id.replace('/', '__')}{suffix}.jsonl"

    done = load_done_keys(out_path, RESUME_KEY)
    pending = [r for r in rows if (model_id, r["image_id"], r["prompt_id"]) not in done]
    log.info(f"{model_id}: {len(done)} done, {len(pending)} pending")
    if not pending:
        return
    if dry_run:
        log.info(f"[dry run] would run {len(pending)} prompts for {model_id}")
        return

    runner = make_runner(model_cfg, defaults)
    runner.load()

    batched = model_cfg.get("backend") == "vllm" and batch_size > 1
    chunks = ([pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
              if batched else [[r] for r in pending])

    started, completed = time.perf_counter(), 0
    for chunk in chunks:
        if batched:
            items = [(r["image_path"], r["prompt_text"], r["system_prompt"]) for r in chunk]
            results = runner.generate_batch(items, max_tokens=model_cfg.get("max_tokens"))
        else:
            row = chunk[0]
            results = [runner.generate(row["image_path"], row["prompt_text"],
                                       row["system_prompt"],
                                       max_tokens=model_cfg.get("max_tokens"))]
        for row, result in zip(chunk, results):
            append_jsonl(out_path, transcript(model_id, row, result))
            completed += 1

        if completed % 100 < len(chunk):
            rate = completed / max(1e-6, time.perf_counter() - started)
            eta = (len(pending) - completed) / rate / 60 if rate else 0
            log.info(f"  {model_id}: {completed}/{len(pending)} ({rate:.1f}/s, ETA {eta:.0f} min)")

    log.info(f"{model_id}: wrote {completed} transcripts -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="config/models.yaml")
    ap.add_argument("--tasks", default="config/tasks.yaml")
    ap.add_argument("--prompts", default="config/prompts.yaml")
    ap.add_argument("--manifest", default="data/exam_set.csv")
    ap.add_argument("--output-dir", default="raw_responses")
    ap.add_argument("--model", default=None, help="run only this model id")
    ap.add_argument("--phrasing", nargs="+", default=["neutral"],
                    choices=["neutral", "terse", "clinician"])
    ap.add_argument("--format", nargs="+", default=["MC", "OE"], choices=["MC", "OE"])
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap prompts, for a smoke test")
    ap.add_argument("--array-index", type=int, default=None, help="SLURM array index picks a model")
    ap.add_argument("--batch-size", type=int, default=8, help="vLLM batch size")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    models_cfg = load_yaml(args.models)
    if args.list_models:
        for model in models_cfg["models"]:
            print(model["id"])
        return

    setup_logging(log_file=f"logs/inference_{(args.model or 'all').replace('/', '__')}.log")
    manifest = pd.read_csv(args.manifest)
    rows = render_all(load_yaml(args.prompts), load_yaml(args.tasks), manifest, args.phrasing)
    rows = [r for r in rows if r["format"] in args.format]
    rows = attach_image_paths(rows, manifest)

    suffix = ""
    if args.num_shards > 1:
        before = len(rows)
        rows = [r for r in rows if shard_of(r, args.num_shards) == args.shard_index]
        suffix = f".shard{args.shard_index}of{args.num_shards}"
        log.info(f"shard {args.shard_index}/{args.num_shards}: {len(rows)}/{before} prompts")
    if args.limit:
        rows = rows[:args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    defaults = models_cfg.get("defaults", {})
    for model_cfg in select_models(models_cfg, args.model, args.array_index):
        log.info(f"=== {model_cfg['id']} ===")
        run_model(model_cfg, defaults, rows, output_dir,
                  args.batch_size, suffix, args.dry_run)


if __name__ == "__main__":
    main()

"""Parse the model transcripts once and write the per-slice decision table.

Everything in Section 4 is computed from the file this writes, so the
transcripts are parsed exactly once and every later number is traceable to one
artifact. The table is small enough to ship with the repository, which is what
lets the tables and figures be rebuilt without the images or the 650 MB of
transcripts.

Usage:
    python -m src.analysis.build_study_table \
        --responses-dir raw_responses \
        --manifest data/exam_set.csv \
        --output results/tumor_decisions.csv.gz
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.analysis import study
from src.utils import get_logger, load_yaml, model_labels, setup_logging, write_csv

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--responses-dir", default="raw_responses")
    ap.add_argument("--manifest", default="data/exam_set.csv")
    ap.add_argument("--tasks", default="config/tasks.yaml")
    ap.add_argument("--models", default="config/models.yaml")
    ap.add_argument("--study", default="config/study.yaml")
    ap.add_argument("--output", default="results/tumor_decisions.csv.gz")
    args = ap.parse_args()

    setup_logging()
    study_cfg = load_yaml(args.study)
    labels = model_labels(args.models)

    records = study.read_transcripts(args.responses_dir, study_cfg["decision"], list(labels))
    if not records:
        raise SystemExit(f"no transcripts found under {args.responses_dir}")

    table = study.build(
        manifest=pd.read_csv(args.manifest),
        records=records,
        tasks_cfg=load_yaml(args.tasks),
        reference_cfg=study_cfg["reference"],
        labels=labels,
    )
    write_csv(table, args.output, compression="gzip")

    log.info(f"{len(table)} rows -> {args.output}")
    log.info("strata (one model):\n" + study.summarize_strata(table).to_string(index=False))
    coverage = (table.groupby("model_label")
                     .agg(slices=("image_id", "size"), answered=("answered", "sum")))
    coverage["answered_share"] = (coverage["answered"] / coverage["slices"]).round(3)
    log.info("answer coverage:\n" + coverage.to_string())


if __name__ == "__main__":
    main()

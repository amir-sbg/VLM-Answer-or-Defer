"""Assemble the per-slice tumor decision table that every result is built on.

One row per (model, slice): the reference, the model's answer, whether that
answer is right, and the stated confidence. Everything in Section 4 reads this
table and nothing else re-parses a transcript.

Two conventions are worth stating plainly, because they decide what the
headline numbers mean.

*Deferred slices stay in the table.* A response with no parseable option, an
UNSURE, a truncated reasoning trace or no parseable confidence is marked
unanswered. It can never be accepted by a gate, but it is still one of the n
slices coverage is measured over, so coverage says how much of the slice task
could be automated rather than how much of the answered part could be.

*Rates use answered slices.* Sensitivity, specificity, the predictive values
and the correlation are computed over answered slices only, since an
unanswered slice has no answer to be right or wrong about.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring.parse import is_reasoning_model, parse_response
from src.utils import get_logger, iter_jsonl, load_yaml, model_labels, project_path

log = get_logger(__name__)

STRATA = ["BraTS+", "BraTS-", "IXI"]
POPULATIONS = ["BraTS", "IXI"]

COLUMNS = [
    "model", "model_label", "image_id", "subject_id", "volume_id", "dataset",
    "modality", "plane", "stratum", "population", "reference", "answer",
    "answered", "correct", "confidence", "lesion_area_px", "mask_distance_mm",
]


def assign_strata(manifest: pd.DataFrame, reference_cfg: dict) -> pd.DataFrame:
    """Label each slice BraTS+, BraTS- or IXI, and drop what cannot be called.

    A BraTS slice is positive when its mask covers at least
    `positivity_threshold_px` pixels on that slice. A slice whose segmentation
    could not be read has no reference at all and is dropped. A slice with a
    readable but sub-threshold mask is either negative or dropped, depending on
    `subthreshold`.
    """
    threshold = int(reference_cfg["positivity_threshold_px"])
    healthy = set(reference_cfg.get("healthy_sources", ["ixi"]))

    frame = manifest.copy()
    area = pd.to_numeric(frame["slice_tumor_area_px"], errors="coerce")

    is_brats = frame["dataset"] == "brats"
    unreadable = is_brats & (area.isna() | (area < 0))
    positive = is_brats & (area >= threshold)
    subthreshold = is_brats & (area >= 1) & (area < threshold)
    zero = is_brats & (area == 0)

    frame["stratum"] = np.where(frame["dataset"].isin(healthy), "IXI", None)
    frame.loc[positive, "stratum"] = "BraTS+"
    frame.loc[zero, "stratum"] = "BraTS-"
    if str(reference_cfg.get("subthreshold", "negative")).lower() == "negative":
        frame.loc[subthreshold, "stratum"] = "BraTS-"
    frame.loc[unreadable, "stratum"] = None

    dropped = int(frame["stratum"].isna().sum())
    if dropped:
        log.info(f"dropped {dropped} slices with no usable reference")

    frame = frame[frame["stratum"].notna()].copy()
    frame["population"] = np.where(frame["stratum"] == "IXI", "IXI", "BraTS")
    frame["reference"] = (frame["stratum"] == "BraTS+").astype(int)
    frame["lesion_area_px"] = area.reindex(frame.index).clip(lower=0)
    return frame


def read_transcripts(responses_dir: str | Path, decision: dict,
                     model_ids: list[str] | None = None) -> list[dict]:
    """Every transcript for the decision under study, across shard files."""
    responses_dir = project_path(responses_dir)
    wanted = set(model_ids) if model_ids else None

    records = []
    for path in sorted(glob.glob(str(responses_dir / "*.jsonl"))):
        for record in iter_jsonl(path):
            if (record.get("task") == decision["task"]
                    and record.get("format") == decision["format"]
                    and record.get("phrasing") == decision["phrasing"]
                    and (wanted is None or record.get("model") in wanted)):
                records.append(record)

    log.info(f"read {len(records)} transcripts for "
             f"{decision['task']}/{decision['format']}/{decision['phrasing']}")
    return records


def parse_decisions(records: list[dict], tasks_cfg: dict) -> pd.DataFrame:
    """Parse transcripts into answers and confidences, one row each."""
    answer_space = [str(a) for a in
                    tasks_cfg["tasks"][records[0]["task"]]["answer_space"]] if records else []

    rows = []
    for record in records:
        parsed = parse_response(
            response_text=record.get("response", ""),
            format_=record.get("format", "MC"),
            answer_space=answer_space,
            mc_options=[str(o) for o in record.get("mc_options", [])],
            reasoning=is_reasoning_model(record.get("model", "")),
        )
        rows.append({
            "model": record["model"],
            "image_id": record["image_id"],
            "answer_raw": parsed["pred_answer"],
            "confidence_raw": parsed["confidence"],
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # The same slice can appear twice if a shard was rerun after a partial
    # write; the transcripts are identical, so keeping the first is safe.
    before = len(frame)
    frame = frame.drop_duplicates(subset=["model", "image_id"], keep="first")
    if len(frame) < before:
        log.info(f"dropped {before - len(frame)} duplicate (model, slice) transcripts")
    return frame


def build(manifest: pd.DataFrame, records: list[dict], tasks_cfg: dict,
          reference_cfg: dict, labels: dict[str, str]) -> pd.DataFrame:
    """Join parsed answers onto the strata to produce the study table."""
    slices = assign_strata(manifest, reference_cfg)
    decisions = parse_decisions(records, tasks_cfg)
    if decisions.empty:
        raise ValueError("no transcripts matched the decision under study")

    frame = decisions.merge(
        slices[["image_id", "subject_id", "volume_id", "dataset", "modality", "plane",
                "stratum", "population", "reference", "lesion_area_px",
                "slice_mask_distance_mm"]],
        on="image_id", how="inner",
    ).rename(columns={"slice_mask_distance_mm": "mask_distance_mm"})

    answer = frame["answer_raw"].map({"yes": 1, "no": 0})
    confidence = pd.to_numeric(frame["confidence_raw"], errors="coerce") / 100.0

    # An answer with no parseable confidence cannot be placed on either side of
    # a threshold, so it counts as deferred just like an unparseable answer.
    frame["answered"] = answer.notna() & confidence.notna()
    frame["answer"] = answer
    frame["confidence"] = confidence
    frame["correct"] = np.where(frame["answered"], answer == frame["reference"], np.nan)
    frame["model_label"] = frame["model"].map(labels).fillna(frame["model"])

    frame = frame[COLUMNS].sort_values(["model_label", "image_id"], kind="mergesort")
    return frame.reset_index(drop=True)


def load(path: str | Path = "results/tumor_decisions.csv.gz") -> pd.DataFrame:
    """Read a study table written by `scripts/build_study_table.py`."""
    frame = pd.read_csv(project_path(path))
    frame["answered"] = frame["answered"].astype(bool)
    return frame


def answered(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows with both an answer and a confidence. Rates are computed on these."""
    return frame[frame["answered"]]


def ordered_models(frame: pd.DataFrame, models_cfg: str = "config/models.yaml") -> list[str]:
    """Model labels present in `frame`, in the order of the paper's tables."""
    order = list(model_labels(models_cfg).values())
    present = set(frame["model_label"])
    return [label for label in order if label in present]


def summarize_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Table 1: the three strata, with their sources, subjects and slice counts."""
    one_model = frame[frame["model_label"] == frame["model_label"].iloc[0]]
    rows = []
    for stratum in STRATA:
        part = one_model[one_model["stratum"] == stratum]
        if part.empty:
            continue
        rows.append({
            "stratum": stratum,
            "source": "BraTS 2024" if stratum.startswith("BraTS") else "IXI",
            "subjects": part["subject_id"].nunique(),
            "sampled_volumes": part["volume_id"].nunique(),
            "slices": len(part),
            "reference": "positive" if stratum == "BraTS+" else "negative",
        })
    return pd.DataFrame(rows)


def load_config(study_path: str | Path = "config/study.yaml") -> dict:
    return load_yaml(study_path)

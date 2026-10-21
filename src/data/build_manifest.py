"""Build the exam manifest: one row per slice, with its reference labels.

The manifest is the single join key between the images, the prompts and the
model transcripts. Everything downstream addresses a slice by `image_id`.

Usage:
    python -m src.data.build_manifest \
        --provenance data/slice_provenance.jsonl \
        --config config/datasets.yaml \
        --study config/study.yaml \
        --output data/exam_set.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.data.slice_labels import compute_slice_labels, healthy_labels
from src.utils import get_logger, load_yaml, read_jsonl, setup_logging, write_csv

log = get_logger(__name__)

COLUMNS = [
    "image_id", "path", "dataset", "subject_id", "volume_id",
    "modality", "plane", "slice_frac",
    "label_modality", "label_plane", "label_tumor_present", "label_laterality",
    "demo_sex", "demo_age", "slice_resolution",
    "slice_tumor_present", "slice_tumor_area_px", "slice_tumor_area_frac",
    "slice_laterality", "slice_mask_distance_mm",
]


def ixi_demographics(meta_path: Path) -> dict[str, dict]:
    """IXI ships age and sex in a spreadsheet keyed by numeric subject id."""
    if not meta_path.exists():
        log.warning(f"IXI metadata not found at {meta_path}; demographics left blank")
        return {}
    try:
        frame = pd.read_excel(meta_path)
    except Exception as e:
        log.warning(f"cannot read IXI metadata: {e}")
        return {}

    frame.columns = [str(c).strip().upper() for c in frame.columns]
    demo: dict[str, dict] = {}
    for _, row in frame.iterrows():
        if pd.isna(row.get("IXI_ID")):
            continue
        subject = f"IXI{int(row['IXI_ID']):03d}"
        sex_code = row.get("SEX_ID")
        demo[subject] = {
            "sex": {1: "M", 2: "F"}.get(sex_code),
            "age": float(row["AGE"]) if not pd.isna(row.get("AGE")) else None,
        }
    return demo


def subject_of(record: dict) -> str:
    """Patient identity behind a sampled volume. This is the bootstrap cluster.

    A BraTS volume id is a sequence filename such as
    ``BraTS-GLI-00027-101-t1c.nii.gz``: the same patient can appear under
    several sequences and several timepoints. Resampling those slices
    independently would treat one patient as several and understate every
    interval, so the cluster is the patient, ``BraTS-GLI-00027``.

    An IXI filename already names the volunteer, so the volume id is returned
    unchanged.
    """
    volume_id = str(record.get("volume_id", record.get("subject_id", "")))
    if record.get("dataset") == "brats":
        match = re.match(r"(BraTS-[A-Za-z]+-\d+)", volume_id)
        return match.group(1) if match else volume_id
    return volume_id


def build(provenance: list[dict], datasets_cfg: dict, positivity_threshold_px: int) -> pd.DataFrame:
    ixi_meta = Path(datasets_cfg["datasets"]["ixi"].get("metadata_file", "data/ixi/IXI.xls"))
    demographics = ixi_demographics(ixi_meta)

    rows = []
    for rec in provenance:
        dataset = rec.get("dataset", "")
        if dataset == "brats":
            labels = compute_slice_labels(
                rec.get("volume_path", ""), rec["plane"], float(rec["slice_frac"]),
                positivity_threshold_px=positivity_threshold_px,
            )
            demo = {"sex": None, "age": None}
        else:
            labels = healthy_labels()
            demo = demographics.get(rec.get("volume_id", ""), {"sex": None, "age": None})

        rows.append({
            "image_id": rec["image_id"],
            "path": rec.get("path"),
            "dataset": dataset,
            "subject_id": subject_of(rec),
            "volume_id": rec.get("volume_id"),
            "modality": rec.get("modality"),
            "plane": rec.get("plane"),
            "slice_frac": rec.get("slice_frac"),
            "label_modality": rec.get("modality"),
            "label_plane": rec.get("plane"),
            "label_tumor_present": labels["slice_tumor_present"],
            "label_laterality": labels["slice_laterality"],
            "demo_sex": demo.get("sex"),
            "demo_age": demo.get("age"),
            "slice_resolution": rec.get("slice_resolution"),
            **labels,
        })

    frame = pd.DataFrame(rows, columns=COLUMNS)
    return frame.dropna(subset=["image_id", "path"]).drop_duplicates(subset=["image_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provenance", default="data/slice_provenance.jsonl")
    ap.add_argument("--config", default="config/datasets.yaml")
    ap.add_argument("--study", default="config/study.yaml")
    ap.add_argument("--output", default="data/exam_set.csv")
    args = ap.parse_args()

    setup_logging(log_file="logs/build_manifest.log")
    provenance = read_jsonl(args.provenance)
    if not provenance:
        raise SystemExit(f"no provenance records in {args.provenance}")

    study = load_yaml(args.study)
    threshold = study["reference"]["positivity_threshold_px"]
    frame = build(provenance, load_yaml(args.config), threshold)
    write_csv(frame, args.output)

    log.info(f"{len(frame)} slices -> {args.output}")
    log.info(f"by dataset:\n{frame['dataset'].value_counts().to_string()}")
    brats = frame[frame.dataset == "brats"]
    log.info(f"BraTS reference at {threshold} px:\n"
             f"{brats['slice_tumor_present'].value_counts().to_string()}")
    log.info(f"sampled volumes: {frame.groupby('dataset')['volume_id'].nunique().to_dict()}")
    log.info(f"patients:        {frame.groupby('dataset')['subject_id'].nunique().to_dict()}")


if __name__ == "__main__":
    main()

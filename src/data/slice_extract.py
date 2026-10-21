"""NIfTI volumes -> 512x512 8-bit PNG slices, with provenance for every slice.

The geometry here is the contract the rest of the repository relies on. A
reference mask is only meaningful if it is cut from the same volume, in the same
orientation, at the same index and through the same letterbox as the image the
model saw, so `reorient_to_ras`, `extract_slice` and `letterbox_resize` are
imported by the label and figure code rather than reimplemented there.

Usage:
    python -m src.data.slice_extract \
        --config config/datasets.yaml \
        --dataset brats ixi \
        --output-dir data/slices \
        --provenance-out data/slice_provenance.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import nibabel as nib
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from src.utils import append_jsonl, get_logger, load_yaml, make_rng, setup_logging

log = get_logger(__name__)

PLANE_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}


class SliceSpec(NamedTuple):
    volume_id: str
    volume_path: Path
    modality: str
    dataset: str
    plane: str
    slice_frac: float


def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Canonical RAS+ orientation, so plane names mean the same thing everywhere."""
    return nib.as_closest_canonical(img)


def intensity_window(data: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Robust percentile clip, then scale to uint8.

    MRI intensities have no fixed unit, and a handful of bright voxels would
    otherwise push the whole brain into the bottom of the range.
    """
    lo, hi = np.percentile(data, p_low), np.percentile(data, p_high)
    if hi == lo:
        return np.zeros_like(data, dtype=np.uint8)
    return (((np.clip(data, lo, hi) - lo) / (hi - lo)) * 255).astype(np.uint8)


def slice_index(shape: tuple[int, ...], plane: str, frac: float) -> int:
    """Index of the slice at `frac` depth along `plane`, clipped to the volume."""
    axis = PLANE_AXIS[plane]
    return int(np.clip(int(round(frac * shape[axis])), 0, shape[axis] - 1))


def extract_slice(volume: np.ndarray, plane: str, frac: float) -> np.ndarray:
    """One 2-D slice from a RAS+ volume."""
    idx = slice_index(volume.shape, plane, frac)
    if plane == "axial":
        return volume[:, :, idx]
    if plane == "sagittal":
        return volume[idx, :, :]
    if plane == "coronal":
        return volume[:, idx, :]
    raise ValueError(f"unknown plane: {plane}")


def letterbox_resize(
    slc: np.ndarray,
    target: tuple[int, int] = (512, 512),
    nearest: bool = False,
) -> np.ndarray:
    """Scale to `target` keeping the aspect ratio, pad the rest with black.

    `nearest` is for binary masks: a mask resampled with Lanczos picks up
    fractional values at the border and no longer outlines the same pixels.
    """
    h, w = slc.shape
    th, tw = target
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    if nearest:
        src = Image.fromarray((slc > 0).astype(np.uint8) * 255, mode="L")
        img = src.resize((nw, nh), Image.NEAREST)
    else:
        img = Image.fromarray(slc, mode="L").resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (tw, th), 0)
    canvas.paste(img, ((tw - nw) // 2, (th - nh) // 2))
    out = np.array(canvas)
    return (out > 0).astype(np.uint8) if nearest else out


def make_image_id(dataset: str, volume_id: str, modality: str, plane: str, frac: float) -> str:
    """Deterministic identifier that carries its own provenance."""
    depth = f"{frac:.2f}".replace(".", "p")
    return f"{dataset}__{volume_id}__{modality}__{plane}__{depth}"


def _sha256_prefix(path: Path, n_bytes: int = 65536) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            h.update(f.read(n_bytes))
    except OSError:
        return "unknown"
    return h.hexdigest()[:16]


def extract_from_volume(
    spec: SliceSpec,
    output_dir: Path,
    target_size: tuple[int, int] = (512, 512),
    clip: tuple[float, float] = (1.0, 99.0),
) -> dict | None:
    """Write one slice as a PNG and return its provenance record."""
    try:
        img = reorient_to_ras(nib.load(str(spec.volume_path)))
    except Exception as e:
        log.error(f"cannot load {spec.volume_path}: {e}")
        return None

    data = img.get_fdata(dtype=np.float32)
    raw = extract_slice(intensity_window(data, *clip), spec.plane, spec.slice_frac)
    image_id = make_image_id(spec.dataset, spec.volume_id, spec.modality,
                             spec.plane, spec.slice_frac)
    record = {
        "image_id": image_id,
        "volume_id": spec.volume_id,
        "dataset": spec.dataset,
        "modality": spec.modality,
        "plane": spec.plane,
        "slice_frac": spec.slice_frac,
        "volume_path": str(spec.volume_path),
        "volume_sha256_prefix": _sha256_prefix(spec.volume_path),
        "volume_shape": list(data.shape),
        "voxel_size_mm": [float(z) for z in img.header.get_zooms()[:3]],
        "slice_resolution": f"{raw.shape[1]}x{raw.shape[0]}",
        "output_size": list(target_size),
        "intensity_clip": list(clip),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_id}.png"
    meta = PngInfo()
    meta.add_text("provenance", json.dumps(record))
    Image.fromarray(letterbox_resize(raw, target_size), mode="L").save(out_path, pnginfo=meta)

    record["path"] = str(out_path.relative_to(Path.cwd())) if out_path.is_absolute() else str(out_path)
    return record


# ── volume discovery ──────────────────────────────────────────────────────────

_BRATS_MODALITY = {"t1n": "T1", "t1c": "T1ce", "t2w": "T2", "t2f": "FLAIR"}
_IXI_MODALITY = {"T1": "T1", "T2": "T2", "PD": "PD", "MRA": "MRA"}


def discover_volumes(root: Path, dataset: str) -> list[tuple[str, Path, str]]:
    """(volume_id, volume_path, modality) for one dataset, sorted for determinism.

    The volume id is the unit the study samples. For BraTS that is one sequence
    file, because each patient folder holds four co-registered sequences and the
    exam shows one slice per sequence; for IXI the filename already names the
    volunteer. The patient behind a BraTS volume is recovered in
    `build_manifest.subject_of`, and that patient is the bootstrap cluster.
    """
    if not root.exists():
        log.warning(f"dataset root missing: {root}")
        return []

    found: list[tuple[str, Path, str]] = []
    for nii in sorted(root.rglob("*.nii.gz")):
        stem = nii.name[: -len(".nii.gz")]
        if dataset == "brats":
            if "seg" in stem.lower():
                continue
            for suffix, modality in _BRATS_MODALITY.items():
                if stem.lower().endswith(suffix):
                    found.append((nii.name, nii, modality))
                    break
        elif dataset == "ixi":
            for key, modality in _IXI_MODALITY.items():
                if f"-{key}" in stem:
                    found.append((stem.split("-")[0], nii, modality))
                    break
        else:
            raise ValueError(f"no discovery rule for dataset: {dataset}")
    return sorted(found, key=lambda v: (v[0], v[2]))


def select_volumes(volumes: list[tuple[str, Path, str]], n_volumes: int, seed: int) -> set[str]:
    """Reproducible subsample of volume ids, drawn once from a sorted list."""
    ids = sorted({v[0] for v in volumes})
    if n_volumes >= len(ids):
        return set(ids)
    picked = make_rng(seed).choice(len(ids), size=n_volumes, replace=False)
    return {ids[i] for i in sorted(picked)}


def run_dataset(name: str, config: dict, output_dir: Path) -> list[dict]:
    cfg = config["datasets"][name]
    glob_cfg = config.get("global", {})
    target = tuple(glob_cfg.get("output_image_size", [512, 512]))
    clip = tuple(glob_cfg.get("intensity_clip_percentile", [1, 99]))

    volumes = discover_volumes(Path(cfg["root"]), name)
    if not volumes:
        return []
    keep = select_volumes(volumes, cfg.get("n_volumes", len(volumes)), cfg.get("subsample_seed", 0))
    volumes = [v for v in volumes if v[0] in keep]
    log.info(f"{name}: {len(keep)} sampled volume ids, {len(volumes)} volumes")

    records = []
    for volume_id, volume_path, modality in volumes:
        for plane in cfg.get("planes", ["axial", "sagittal", "coronal"]):
            for frac in cfg.get("slice_fracs", [0.4, 0.5, 0.6]):
                spec = SliceSpec(volume_id, volume_path, modality, name, plane, float(frac))
                rec = extract_from_volume(spec, output_dir / name, target, clip)
                if rec is not None:
                    records.append(rec)
    log.info(f"{name}: wrote {len(records)} slices")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/datasets.yaml")
    ap.add_argument("--dataset", nargs="+", default=["brats", "ixi"])
    ap.add_argument("--output-dir", default="data/slices")
    ap.add_argument("--provenance-out", default="data/slice_provenance.jsonl")
    args = ap.parse_args()

    setup_logging(log_file="logs/slice_extract.log")
    config = load_yaml(args.config)

    records: list[dict] = []
    for name in args.dataset:
        records.extend(run_dataset(name, config, Path(args.output_dir)))

    out = Path(args.provenance_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    for rec in records:
        append_jsonl(out, rec)
    log.info(f"{len(records)} slices total; provenance -> {out}")


if __name__ == "__main__":
    main()

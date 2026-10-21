"""Per-slice reference labels derived from the BraTS segmentation mask.

The reference is a property of the *slice*, not of the patient. Most 2-D slices
through a glioma brain contain no annotated tissue, so labelling every slice of
a tumor subject positive would score a model wrong for correctly answering "no"
on a tumor-free slice.

Every quantity here is cut from the segmentation volume with the same
reorientation and the same index formula used to write the PNG
(`src.data.slice_extract`), so the mask lines up with the image the model saw.

Fields produced per BraTS slice:
    slice_tumor_present     "yes" | "no"   mask area >= positivity threshold
    slice_tumor_area_px     int            annotated pixels on this slice
    slice_tumor_area_frac   float          that count over the slice size
    slice_laterality        left | right | bilateral | none
    slice_mask_distance_mm  float          for mask-free slices, how far the
                                           nearest annotated voxel is

`slice_mask_distance_mm` is what the distant-negative subset of Section 4.6 is
built from. The minimum distance from a whole slice plane to the mask is the
out-of-plane offset to the nearest annotated voxel: the in-plane position of
that voxel is itself inside the plane, so no in-plane term survives the minimum.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from src.data.slice_extract import PLANE_AXIS, extract_slice, reorient_to_ras, slice_index
from src.utils import get_logger

log = get_logger(__name__)

# One mask per subject is read dozens of times (4 sequences x 3 planes x 3 depths).
_SEG_CACHE: dict[str, tuple[np.ndarray, tuple[float, float, float]] | None] = {}

# Share of annotated mass in one hemisphere above which a slice counts as lateralized.
LATERALITY_THRESHOLD = 0.60


def find_seg_mask(volume_path: Path) -> Path | None:
    """The segmentation that goes with a BraTS modality volume.

    BraTS ships one mask per subject folder next to the four sequences, so
    matching on the folder is more robust than rewriting the filename suffix.
    """
    for pattern in ("*seg*.nii.gz", "*seg*.nii"):
        matches = sorted(volume_path.parent.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_segmentation(volume_path: str) -> tuple[np.ndarray, tuple[float, float, float]] | None:
    """Load the RAS+ segmentation and its voxel sizes, or None if unavailable."""
    if volume_path in _SEG_CACHE:
        return _SEG_CACHE[volume_path]

    seg_path = find_seg_mask(Path(volume_path))
    result = None
    if seg_path is not None:
        try:
            img = reorient_to_ras(nib.load(str(seg_path)))
            result = (img.get_fdata(dtype=np.float32) > 0,
                      tuple(float(z) for z in img.header.get_zooms()[:3]))
        except Exception as e:
            log.warning(f"cannot load segmentation {seg_path}: {e}")
    _SEG_CACHE[volume_path] = result
    return result


def mask_slice(volume_path: str, plane: str, frac: float) -> np.ndarray | None:
    """The binary mask slice aligned with the extracted PNG, before letterboxing."""
    loaded = load_segmentation(volume_path)
    if loaded is None:
        return None
    return extract_slice(loaded[0], plane, frac)


def distance_to_mask_mm(mask: np.ndarray, spacing: tuple[float, float, float],
                        plane: str, idx: int) -> float:
    """Out-of-plane distance from slice `idx` to the nearest annotated voxel."""
    axis = PLANE_AXIS[plane]
    annotated = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
    if annotated.size == 0:
        return float("inf")
    return float(np.abs(annotated - idx).min() * spacing[axis])


def slice_laterality(mask_2d: np.ndarray, plane: str, volume_shape: tuple[int, ...],
                     idx: int) -> str:
    """Which side of the brain the annotated tissue sits on.

    In RAS+, x increases to the right. Axial and coronal slices keep x as their
    first in-plane axis. A sagittal slice has no in-plane left/right axis at all:
    the whole slice sits at one x, so its side is the side of that plane.
    """
    if not mask_2d.any():
        return "none"
    if plane == "sagittal":
        return "right" if idx > volume_shape[0] / 2.0 else "left"
    xs = np.nonzero(mask_2d)[0]
    midline = mask_2d.shape[0] / 2.0
    frac_right = (xs > midline).mean()
    if frac_right >= LATERALITY_THRESHOLD:
        return "right"
    if (1.0 - frac_right) >= LATERALITY_THRESHOLD:
        return "left"
    return "bilateral"


def unknown_labels() -> dict[str, Any]:
    return {
        "slice_tumor_present": "unknown",
        "slice_tumor_area_px": -1,
        "slice_tumor_area_frac": float("nan"),
        "slice_laterality": "unknown",
        "slice_mask_distance_mm": float("nan"),
    }


def healthy_labels() -> dict[str, Any]:
    """Labels for a slice from a healthy-volunteer volume, which carries no mask."""
    return {
        "slice_tumor_present": "no",
        "slice_tumor_area_px": 0,
        "slice_tumor_area_frac": 0.0,
        "slice_laterality": "none",
        "slice_mask_distance_mm": float("inf"),
    }


def compute_slice_labels(volume_path: str, plane: str, frac: float,
                         positivity_threshold_px: int = 25) -> dict[str, Any]:
    """Reference labels for one BraTS slice.

    `slice_tumor_area_px` is always the raw annotated area; the threshold only
    decides `slice_tumor_present`. Keeping the raw count is what lets the
    analysis drop the ambiguous 1..24 px slices and re-run Section 4.6 at a
    different threshold without touching the volumes again.
    """
    loaded = load_segmentation(volume_path)
    if loaded is None:
        return unknown_labels()
    mask, spacing = loaded

    idx = slice_index(mask.shape, plane, frac)
    mask_2d = extract_slice(mask, plane, frac)
    area_px = int(mask_2d.sum())
    present = "yes" if area_px >= positivity_threshold_px else "no"

    return {
        "slice_tumor_present": present,
        "slice_tumor_area_px": area_px,
        "slice_tumor_area_frac": round(area_px / float(mask_2d.size), 6) if mask_2d.size else 0.0,
        "slice_laterality": slice_laterality(mask_2d, plane, mask.shape, idx)
                            if present == "yes" else "none",
        "slice_mask_distance_mm": distance_to_mask_mm(mask, spacing, plane, idx),
    }

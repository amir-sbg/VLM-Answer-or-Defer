"""Shared fixtures: a tiny synthetic study table with known answers.

The fixtures build a table by hand rather than loading the real one, so a test
that fails points at the code rather than at the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def decision(model: str, image: str, stratum: str, reference: int, answer: int | None,
             confidence: float | None, subject: str = "S1", area: float = 0.0,
             distance: float = float("inf"), modality: str = "T2",
             plane: str = "axial") -> dict:
    answered = answer is not None and confidence is not None
    return {
        "model": f"org/{model}", "model_label": model, "image_id": image,
        "subject_id": subject, "volume_id": subject, "dataset":
            "ixi" if stratum == "IXI" else "brats",
        "modality": modality, "plane": plane, "stratum": stratum,
        "population": "IXI" if stratum == "IXI" else "BraTS",
        "reference": reference, "answer": answer, "answered": answered,
        "correct": float(answer == reference) if answered else float("nan"),
        "confidence": confidence, "lesion_area_px": area, "mask_distance_mm": distance,
    }


@pytest.fixture
def simple_table() -> pd.DataFrame:
    """Ten slices for one model: a clean confidence ordering plus one deferral.

    Confidence 1.00 holds two correct answers, 0.90 holds three correct and one
    wrong, 0.50 holds three wrong, and one slice is never answered.
    """
    rows = [
        decision("M", "a1", "BraTS+", 1, 1, 1.00, area=500),
        decision("M", "a2", "BraTS+", 1, 1, 1.00, area=900),
        decision("M", "b1", "BraTS+", 1, 1, 0.90, area=300),
        decision("M", "b2", "BraTS+", 1, 1, 0.90, area=200),
        decision("M", "b3", "BraTS-", 0, 0, 0.90, distance=20.0),
        decision("M", "b4", "BraTS-", 0, 1, 0.90, distance=20.0),
        decision("M", "c1", "BraTS-", 0, 1, 0.50, distance=2.0),
        decision("M", "c2", "BraTS-", 0, 1, 0.50, distance=2.0),
        decision("M", "c3", "BraTS+", 1, 0, 0.50, area=40),
        decision("M", "d1", "BraTS+", 1, None, None, area=60),
    ]
    return pd.DataFrame(rows)

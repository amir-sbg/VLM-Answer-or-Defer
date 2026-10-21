"""Assembling the study table: strata, deferral, and the quantities on top."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import rates, study

REFERENCE = {"positivity_threshold_px": 25, "subthreshold": "negative",
             "drop_without_mask": True, "healthy_sources": ["ixi"]}


def manifest_row(image_id: str, dataset: str, area: float) -> dict:
    return {"image_id": image_id, "dataset": dataset, "slice_tumor_area_px": area,
            "subject_id": "S", "volume_id": "V", "modality": "T2", "plane": "axial",
            "slice_mask_distance_mm": 0.0}


@pytest.fixture
def manifest() -> pd.DataFrame:
    return pd.DataFrame([
        manifest_row("big", "brats", 500),      # positive
        manifest_row("edge", "brats", 25),      # positive, exactly at the threshold
        manifest_row("tiny", "brats", 7),       # sub-threshold
        manifest_row("clean", "brats", 0),      # negative
        manifest_row("nomask", "brats", -1),    # no usable reference
        manifest_row("healthy", "ixi", 0),      # negative
    ])


def test_the_threshold_is_inclusive(manifest):
    strata = study.assign_strata(manifest, REFERENCE).set_index("image_id")["stratum"]
    assert strata["big"] == "BraTS+" and strata["edge"] == "BraTS+"


def test_a_slice_without_a_usable_mask_is_dropped(manifest):
    assert "nomask" not in set(study.assign_strata(manifest, REFERENCE)["image_id"])


def test_sub_threshold_slices_follow_the_configured_rule(manifest):
    as_negative = study.assign_strata(manifest, REFERENCE).set_index("image_id")
    assert as_negative.loc["tiny", "stratum"] == "BraTS-"

    dropped = study.assign_strata(manifest, {**REFERENCE, "subthreshold": "drop"})
    assert "tiny" not in set(dropped["image_id"])


def test_healthy_volunteers_are_their_own_population(manifest):
    strata = study.assign_strata(manifest, REFERENCE).set_index("image_id")
    assert strata.loc["healthy", "stratum"] == "IXI"
    assert strata.loc["healthy", "population"] == "IXI"
    assert strata.loc["clean", "population"] == "BraTS"


def test_rates_ignore_deferred_slices_but_coverage_does_not(simple_table):
    answered = study.answered(simple_table)
    assert len(simple_table) == 10 and len(answered) == 9


def test_sensitivity_and_specificity_match_the_table(simple_table):
    row = rates.by_stratum(simple_table).iloc[0]
    # Answered BraTS+ slices: a1 a2 b1 b2 say yes, c3 says no -> 4/5.
    assert row["sensitivity_brats_pos"] == pytest.approx(0.8)
    # Answered BraTS- slices: b3 says no, b4 c1 c2 say yes -> 1/4.
    assert row["specificity_brats_neg"] == pytest.approx(0.25)
    assert row["balanced_accuracy_brats"] == pytest.approx((0.8 + 0.25) / 2)


def test_the_more_confident_answer_is_the_one_with_the_higher_mean(simple_table):
    summary = rates.answer_summary(study.answered(simple_table))
    assert summary["mu_yes"] > summary["mu_no"]
    assert rates.more_confident_answer(summary) == "yes"


def test_a_constant_confidence_leaves_the_correlation_undefined():
    frame = pd.Series([0.9, 0.9, 0.9]), pd.Series([1, 0, 1])
    assert np.isnan(rates.correlation(*frame))

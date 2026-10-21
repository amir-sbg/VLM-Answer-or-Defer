"""The restricted subset and the subject-clustered bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import bootstrap, subset
from tests.conftest import decision

SUBSET_CFG = {"min_lesion_px": 400, "lesion_sequences": ["T2", "FLAIR", "T1ce"],
              "restrict_negatives_to_sequences": True,
              "min_negative_distance_mm": 10, "reweight_to_pi": 0.70}


@pytest.fixture
def mixed_table() -> pd.DataFrame:
    return pd.DataFrame([
        decision("M", "keep_pos", "BraTS+", 1, 1, 0.9, area=800, modality="T2"),
        decision("M", "small_pos", "BraTS+", 1, 1, 0.9, area=100, modality="T2"),
        decision("M", "t1_pos", "BraTS+", 1, 1, 0.9, area=800, modality="T1"),
        decision("M", "keep_neg", "BraTS-", 0, 0, 0.9, distance=25.0, modality="FLAIR"),
        decision("M", "near_neg", "BraTS-", 0, 0, 0.9, distance=4.0, modality="FLAIR"),
        decision("M", "healthy", "IXI", 0, 0, 0.9),
    ])


def test_the_subset_keeps_only_large_lesions_on_the_named_sequences(mixed_table):
    kept = set(subset.select(mixed_table, SUBSET_CFG)["image_id"])
    assert kept == {"keep_pos", "keep_neg"}


def test_the_sequence_restriction_on_negatives_is_configurable(mixed_table):
    relaxed = {**SUBSET_CFG, "restrict_negatives_to_sequences": False}
    kept = subset.select(mixed_table, relaxed)
    assert (kept["modality"] == "FLAIR").any()


def test_healthy_volunteers_never_enter_the_subset(mixed_table):
    assert "healthy" not in set(subset.select(mixed_table, SUBSET_CFG)["image_id"])


def test_the_answer_only_rule_is_evaluated_at_the_reweighted_mix():
    # sensitivity .8, specificity .6, pi .5: "yes" covers .5*.8 + .5*.4 = .6
    # and is right .4/.6 = .667 of the time, so a 40% budget admits it.
    result = subset.answer_only_at_pi(0.8, 0.6, 0.5, 0.40)
    assert result["rule"] == "all_yes"
    assert result["coverage"] == pytest.approx(0.6)
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_no_rule_is_reported_when_neither_class_meets_the_budget():
    assert subset.answer_only_at_pi(0.5, 0.5, 0.5, 0.10)["rule"] == "none"


# ── bootstrap ─────────────────────────────────────────────────────────────────

def clustered_frame(n_subjects: int = 20, slices: int = 9) -> pd.DataFrame:
    """Each subject answers identically on all of their slices."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(n_subjects):
        answer = int(rng.integers(0, 2))
        for i in range(slices):
            rows.append(decision("M", f"s{s}_{i}", "BraTS+", 1, answer, 0.9,
                                 subject=f"P{s}"))
    return pd.DataFrame(rows)


def test_the_interval_brackets_the_point_estimate():
    cfg = {"n_resamples": 300, "seed": 1, "cluster": "subject_id", "level": 0.95}
    result = bootstrap.interval(clustered_frame(), bootstrap.sensitivity, cfg)
    assert result["low"] <= result["estimate"] <= result["high"]


def test_resampling_subjects_is_wider_than_resampling_slices():
    """The point of clustering: ignoring it understates the uncertainty."""
    frame = clustered_frame()
    by_subject = {"n_resamples": 400, "seed": 2, "cluster": "subject_id", "level": 0.95}
    by_slice = {**by_subject, "cluster": "image_id"}

    wide = bootstrap.interval(frame, bootstrap.sensitivity, by_subject)
    narrow = bootstrap.interval(frame, bootstrap.sensitivity, by_slice)
    assert (wide["high"] - wide["low"]) > (narrow["high"] - narrow["low"])


def test_the_bootstrap_is_reproducible():
    frame = clustered_frame()
    cfg = {"n_resamples": 200, "seed": 7, "cluster": "subject_id", "level": 0.95}
    first = bootstrap.interval(frame, bootstrap.sensitivity, cfg)
    second = bootstrap.interval(frame, bootstrap.sensitivity, cfg)
    assert first == second

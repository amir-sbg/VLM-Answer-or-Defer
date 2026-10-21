"""The gate: coverage, selective risk, C_alpha, and the answer-only baseline."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import gate


def test_operating_points_take_whole_confidence_levels(simple_table):
    points = gate.operating_points(simple_table)
    # Three distinct confidences, so three operating points, highest first.
    assert list(points["threshold"]) == [1.00, 0.90, 0.50]
    assert list(points["accepted"]) == [2, 6, 9]
    assert list(points["errors"]) == [0, 1, 4]


def test_coverage_counts_the_deferred_slice(simple_table):
    points = gate.operating_points(simple_table)
    # Ten slices, nine of them answered: full acceptance still covers 0.9.
    assert points["coverage"].iloc[-1] == pytest.approx(0.9)


def test_c_alpha_picks_the_largest_feasible_level(simple_table):
    strict = gate.max_coverage_at_risk(simple_table, 0.05)
    assert strict["accepted"] == 2 and strict["coverage"] == pytest.approx(0.2)

    loose = gate.max_coverage_at_risk(simple_table, 0.20)
    # Risk at the 0.90 level is 1/6 = 0.167, which fits inside a 20% budget.
    assert loose["accepted"] == 6 and loose["risk"] == pytest.approx(1 / 6)


def test_a_zero_budget_accepts_only_an_error_free_level(simple_table):
    perfect = gate.max_coverage_at_risk(simple_table, 0.0)
    assert perfect["accepted"] == 2 and perfect["risk"] == 0.0


def test_c_alpha_is_zero_when_even_the_top_level_has_an_error():
    frame = pd.DataFrame([{"answered": True, "confidence": c, "correct": z, "answer": 1}
                          for c, z in [(1.0, 0.0), (0.9, 1.0), (0.9, 1.0)]])
    assert gate.max_coverage_at_risk(frame, 0.10)["coverage"] == 0.0


def test_a_threshold_cannot_split_a_tie():
    """Four slices on one confidence, two wrong: no threshold takes only the good ones."""
    rows = [{"answered": True, "confidence": 0.9, "correct": c, "answer": 1}
            for c in (1.0, 1.0, 0.0, 0.0)]
    frame = pd.DataFrame(rows)
    assert gate.max_coverage_at_risk(frame, 0.10)["coverage"] == 0.0
    assert gate.max_coverage_at_risk(frame, 0.50)["accepted"] == 4


def test_answer_only_rule_prefers_the_largest_class_that_fits(simple_table):
    # "yes" is said on 7 of 10 slices and is wrong 3 times (risk 3/7 = 0.429);
    # "no" on 2 and wrong once (0.5); all 9 answers together are wrong 4 times
    # (0.444). A budget that admits only "yes" takes it.
    assert gate.answer_only_rule(simple_table, 0.43)["rule"] == "all_yes"
    assert gate.answer_only_rule(simple_table, 0.43)["coverage"] == pytest.approx(0.7)

    # A budget that also admits the whole answered set prefers it, being larger.
    looser = gate.answer_only_rule(simple_table, 0.45)
    assert looser["rule"] == "all_answers" and looser["coverage"] == pytest.approx(0.9)


def test_answer_only_rule_reports_nothing_when_no_class_fits(simple_table):
    assert gate.answer_only_rule(simple_table, 0.40)["rule"] == "none"


def test_specificity_needed_inverts_the_predictive_value():
    # At pi = 0.5 and sensitivity 1, a 5% error budget needs specificity
    # 1 - (0.05/0.95) = 0.947.
    assert gate.specificity_needed(1.0, 0.5, 0.05) == pytest.approx(1 - 0.05 / 0.95)
    assert gate.specificity_needed(0.0, 0.7, 0.05) == pytest.approx(1.0)


def test_errors_in_the_accepted_set_are_split_by_type(simple_table):
    errors = gate.errors_in_accepted_set(simple_table, 0.20)
    assert errors["accepted"] == 6
    assert errors["false_positives"] == 1 and errors["false_negatives"] == 0

"""Lemma 1 and its corollary."""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis import decomposition
from tests.conftest import decision


def test_the_three_terms_add_up_to_the_total(simple_table):
    result = decomposition.decompose(simple_table)
    assert result["residual"] == pytest.approx(0.0, abs=1e-12)
    assert (result["answer_term"] + result["within_term"]
            == pytest.approx(result["cov_total"], abs=1e-12))


def test_within_answer_terms_vanish_when_every_slice_shares_a_label():
    """Corollary 1(a): on a single-label population, z is fixed by the answer."""
    rows = [decision("M", f"i{i}", "IXI", 0, answer, confidence)
            for i, (answer, confidence) in enumerate(
                [(1, 0.95), (1, 0.80), (1, 0.60), (0, 0.70), (0, 0.50), (0, 0.90)])]
    result = decomposition.decompose(pd.DataFrame(rows))

    assert result["within_yes"] == pytest.approx(0.0, abs=1e-12)
    assert result["within_no"] == pytest.approx(0.0, abs=1e-12)
    assert result["answer_term"] == pytest.approx(result["cov_total"], abs=1e-12)


def test_the_answer_term_sign_follows_the_favoured_and_the_reliable_answer():
    """Positive when the more-confident answer is also the more reliable one."""
    confident_and_right = pd.DataFrame(
        [decision("M", "p1", "BraTS+", 1, 1, 0.95),
         decision("M", "p2", "BraTS+", 1, 1, 0.95),
         decision("M", "n1", "BraTS-", 0, 0, 0.60),
         decision("M", "n2", "BraTS+", 1, 0, 0.60)])
    assert decomposition.decompose(confident_and_right)["answer_term"] > 0

    # Same confidences, but now the confident answer is the unreliable one.
    confident_and_wrong = pd.DataFrame(
        [decision("M", "p1", "BraTS-", 0, 1, 0.95),
         decision("M", "p2", "BraTS-", 0, 1, 0.95),
         decision("M", "n1", "BraTS-", 0, 0, 0.60),
         decision("M", "n2", "BraTS-", 0, 0, 0.60)])
    assert decomposition.decompose(confident_and_wrong)["answer_term"] < 0


def test_a_model_that_gives_one_answer_has_no_answer_term():
    rows = [decision("M", f"y{i}", "BraTS+", 1, 1, c) for i, c in enumerate([0.9, 0.8])]
    rows += [decision("M", "y2", "BraTS-", 0, 1, 0.7)]
    result = decomposition.decompose(pd.DataFrame(rows))
    assert result["answer_term"] == 0.0
    assert result["cov_total"] == pytest.approx(result["within_term"], abs=1e-12)

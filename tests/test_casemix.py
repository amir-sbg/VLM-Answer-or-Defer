"""Reweighting the case mix under label shift."""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis import casemix


def test_predictive_values_at_a_balanced_mix():
    ppv, npv = casemix.predictive_values(0.8, 0.6, 0.5)
    # PPV = .5*.8 / (.5*.8 + .5*.4) = .4/.6
    assert float(ppv) == pytest.approx(0.4 / 0.6)
    assert float(npv) == pytest.approx(0.3 / 0.4)


def test_ppv_rises_and_npv_falls_as_positives_become_common():
    grid = np.linspace(0.05, 0.95, 19)
    ppv, npv = casemix.predictive_values(0.7, 0.7, grid)
    assert np.all(np.diff(ppv) > 0)
    assert np.all(np.diff(npv) < 0)


def test_a_perfect_test_is_right_at_every_mix():
    ppv, npv = casemix.predictive_values(1.0, 1.0, np.linspace(0.1, 0.9, 9))
    assert np.allclose(ppv, 1.0) and np.allclose(npv, 1.0)


def test_predictive_values_cross_at_the_balanced_point_for_a_symmetric_model():
    """With sensitivity equal to specificity, PPV meets NPV at pi = 0.5."""
    assert casemix.crossing_pi(0.75, 0.75) == pytest.approx(0.5, abs=1e-3)


def test_the_crossing_follows_the_rate_the_model_is_worse_at():
    """A lopsided model needs the mix to lean its weak way before the two meet.

    With sensitivity .9 and specificity .5, a balanced mix still leaves NPV
    above PPV -- the negatives it does catch are reliable, while half the
    negatives land in the "yes" pile. Positives have to become common before
    "yes" becomes as trustworthy as "no", so the crossing sits above 0.5.
    """
    assert casemix.crossing_pi(0.9, 0.5) > 0.5
    assert casemix.crossing_pi(0.5, 0.9) < 0.5


def test_the_gemma_style_swing_reproduces_the_reported_magnitude():
    """Sensitivity .616 and specificity .702, as measured on BraTS for Gemma-4-12B."""
    ppv_natural, _ = casemix.predictive_values(0.616, 0.702, 0.697)
    ppv_rare, _ = casemix.predictive_values(0.616, 0.702, 0.10)
    assert 1 - float(ppv_natural) == pytest.approx(0.17, abs=0.01)
    assert 1 - float(ppv_rare) == pytest.approx(0.81, abs=0.01)

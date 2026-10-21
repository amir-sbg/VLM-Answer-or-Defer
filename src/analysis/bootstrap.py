"""Subject-clustered bootstrap intervals.

Nine slices come from each sampled volume and several volumes can come from one
patient, so slices are not independent: a model that misreads one patient's
anatomy misreads all of their slices together. Resampling slices would treat
those as independent observations and return intervals that are too narrow.
The resampling unit here is the patient, and a resampled patient contributes
all of their slices.

The percentile interval is used rather than a normal approximation, because
several of the quantities (a specificity near 1, a correlation near 0) sit
close to a boundary where the sampling distribution is visibly skewed.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

Statistic = Callable[[pd.DataFrame], float]


def clustered_resamples(frame: pd.DataFrame, statistic: Statistic, n_resamples: int,
                        seed: int, cluster: str = "subject_id") -> np.ndarray:
    """Values of `statistic` over `n_resamples` patient-level resamples.

    Each resample is assembled by taking row positions rather than concatenating
    per-patient frames: with 138 patients and 1,000 resamples the concatenation
    dominates the runtime, and one fancy-index take gives the same rows.
    """
    positions = [np.flatnonzero(frame[cluster].to_numpy() == value)
                 for value in pd.unique(frame[cluster])]
    if not positions:
        return np.array([])

    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        picks = rng.integers(0, len(positions), size=len(positions))
        rows = frame.take(np.concatenate([positions[j] for j in picks]))
        try:
            values[i] = statistic(rows)
        except (ValueError, ZeroDivisionError):
            values[i] = np.nan
    return values


def interval(frame: pd.DataFrame, statistic: Statistic, bootstrap_cfg: dict) -> dict[str, float]:
    """Point estimate and percentile interval for one statistic on one frame."""
    level = float(bootstrap_cfg.get("level", 0.95))
    tail = (1.0 - level) / 2.0

    point = statistic(frame)
    values = clustered_resamples(
        frame, statistic,
        n_resamples=int(bootstrap_cfg["n_resamples"]),
        seed=int(bootstrap_cfg["seed"]),
        cluster=str(bootstrap_cfg.get("cluster", "subject_id")),
    )
    usable = values[np.isfinite(values)]
    if usable.size < 2:
        return {"estimate": point, "low": float("nan"), "high": float("nan"), "n_usable": 0}

    low, high = np.quantile(usable, [tail, 1.0 - tail])
    return {"estimate": float(point), "low": float(low), "high": float(high),
            "n_usable": int(usable.size)}


# ── the statistics the paper reports intervals for ────────────────────────────

def _rate(frame: pd.DataFrame, stratum: str, column: str, complement: bool = False) -> float:
    rows = frame[(frame["stratum"] == stratum) & frame["answered"]]
    if rows.empty:
        return float("nan")
    value = float(rows[column].mean())
    return 1.0 - value if complement else value


def sensitivity(frame: pd.DataFrame) -> float:
    return _rate(frame, "BraTS+", "answer")


def specificity_brats(frame: pd.DataFrame) -> float:
    return _rate(frame, "BraTS-", "answer", complement=True)


def specificity_ixi(frame: pd.DataFrame) -> float:
    return _rate(frame, "IXI", "answer", complement=True)


def balanced_accuracy_brats(frame: pd.DataFrame) -> float:
    return (sensitivity(frame) + specificity_brats(frame)) / 2.0


def correlation_on(population: str) -> Statistic:
    """rho between confidence and correctness on one population, as a statistic."""
    from src.analysis.rates import correlation

    def statistic(frame: pd.DataFrame) -> float:
        rows = frame[(frame["population"] == population) & frame["answered"]]
        return correlation(rows["confidence"], rows["correct"])

    return statistic


RATE_STATISTICS: dict[str, Statistic] = {
    "sensitivity_brats_pos": sensitivity,
    "specificity_brats_neg": specificity_brats,
    "specificity_ixi": specificity_ixi,
    "balanced_accuracy_brats": balanced_accuracy_brats,
}


def rate_intervals(table: pd.DataFrame, bootstrap_cfg: dict,
                   statistics: dict[str, Statistic] | None = None) -> pd.DataFrame:
    """Intervals for every rate in Table 2, one row per model and statistic."""
    statistics = statistics or RATE_STATISTICS

    rows = []
    for label, model_rows in table.groupby("model_label", sort=False):
        for name, statistic in statistics.items():
            rows.append({"model": label, "statistic": name,
                         **interval(model_rows, statistic, bootstrap_cfg)})
    return pd.DataFrame(rows)

"""How the reliability of a model's answers moves with the case mix.

The sign flip in Section 4.3 compares two sources that differ in more than
their case mix -- scanners, sequences, skull-stripping. To isolate the mix, we
reweight within BraTS: sensitivity and specificity are held at their measured
values and only the share of reference-positive slices, pi, changes. By Bayes'
rule,

    PPV(pi) = pi*sens / [pi*sens + (1-pi)(1-spec)]
    NPV(pi) = (1-pi)*spec / [(1-pi)*spec + pi*(1-sens)]

This is label shift [Saerens et al., 2002; Lipton et al., 2018]: the model is
unchanged and only the prior moves. It explains why the reliability of a
model's favoured answers -- and with it what a gate can keep -- changes with
the population. It does not test a threshold carried from one case mix to
another, and it does not show that the BraTS-IXI difference is case mix alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.rates import answer_summary, more_confident_answer
from src.analysis.study import answered


def predictive_values(sensitivity: float, specificity: float,
                      pi: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """PPV and NPV at case mix pi, for fixed sensitivity and specificity."""
    pi = np.asarray(pi, dtype=float)
    true_pos = pi * sensitivity
    false_pos = (1.0 - pi) * (1.0 - specificity)
    true_neg = (1.0 - pi) * specificity
    false_neg = pi * (1.0 - sensitivity)

    with np.errstate(invalid="ignore", divide="ignore"):
        ppv = np.where(true_pos + false_pos > 0, true_pos / (true_pos + false_pos), np.nan)
        npv = np.where(true_neg + false_neg > 0, true_neg / (true_neg + false_neg), np.nan)
    return ppv, npv


def crossing_pi(sensitivity: float, specificity: float) -> float:
    """The case mix at which PPV equals NPV.

    The answer term of Lemma 1 changes sign here, provided the model's
    more-confident answer stays the same. Solved on a dense grid rather than in
    closed form: the equation is a quadratic whose root has to be checked
    against the unit interval anyway, and the grid keeps the degenerate cases
    (a model that never says "yes") returning NaN instead of a spurious root.
    """
    grid = np.linspace(1e-6, 1 - 1e-6, 200_001)
    ppv, npv = predictive_values(sensitivity, specificity, grid)
    gap = ppv - npv
    usable = np.isfinite(gap)
    if usable.sum() < 2 or np.nanmin(gap[usable]) > 0 or np.nanmax(gap[usable]) < 0:
        return float("nan")
    return float(grid[usable][np.argmin(np.abs(gap[usable]))])


def measured_rates(frame: pd.DataFrame) -> dict[str, float]:
    """Sensitivity, specificity, natural case mix and favoured answer, within BraTS."""
    rows = answered(frame[frame["population"] == "BraTS"])
    positives = rows[rows["reference"] == 1]
    negatives = rows[rows["reference"] == 0]
    summary = answer_summary(rows)
    return {
        "sensitivity": float(positives["answer"].mean()) if len(positives) else float("nan"),
        "specificity": 1.0 - float(negatives["answer"].mean()) if len(negatives) else float("nan"),
        "natural_pi": len(positives) / len(rows) if len(rows) else float("nan"),
        "more_confident_answer": more_confident_answer(summary),
    }


def pi_grid(case_mix_cfg: dict) -> np.ndarray:
    grid = case_mix_cfg["pi_grid"]
    return np.arange(grid["start"], grid["stop"] + grid["step"] / 2, grid["step"])


def error_curves(table: pd.DataFrame, case_mix_cfg: dict) -> pd.DataFrame:
    """Figure 4: error rate of each model's more-confident answer class against pi."""
    grid = pi_grid(case_mix_cfg)

    frames = []
    for label, model_rows in table.groupby("model_label", sort=False):
        rates = measured_rates(model_rows)
        ppv, npv = predictive_values(rates["sensitivity"], rates["specificity"], grid)
        favoured = rates["more_confident_answer"]
        frames.append(pd.DataFrame({
            "model": label,
            "pi": grid,
            "more_confident_answer": favoured,
            "error_rate": 1.0 - (ppv if favoured == "yes" else npv),
        }))
    return pd.concat(frames, ignore_index=True)


def summary_table(table: pd.DataFrame, case_mix_cfg: dict,
                  reference_pis: tuple[float, ...] = (0.10, 0.50, 0.90)) -> pd.DataFrame:
    """Error rate of the favoured answer at the natural mix and at a few others."""
    rows = []
    for label, model_rows in table.groupby("model_label", sort=False):
        rates = measured_rates(model_rows)
        record = {"model": label, **rates,
                  "crossing_pi": crossing_pi(rates["sensitivity"], rates["specificity"])}

        for pi in (rates["natural_pi"],) + reference_pis:
            ppv, npv = predictive_values(rates["sensitivity"], rates["specificity"], pi)
            error = float(1.0 - (ppv if rates["more_confident_answer"] == "yes" else npv))
            key = "error_at_natural_pi" if pi == rates["natural_pi"] else f"error_at_pi_{pi:.2f}"
            record[key] = error
        rows.append(record)
    return pd.DataFrame(rows)

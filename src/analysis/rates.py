"""Per-model rates on the tumor decision: Table 2 and the quantities Lemma 1 needs.

All rates are computed over answered slices. The share of answers that are
"yes" (q), the predictive values (PPV, NPV) and the mean confidence on each
answer (mu1, mu0) are measured on whichever population they are reported for,
because every one of them moves with the population -- which is the point of
Sections 4.3 and 4.4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.study import answered


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def correlation(confidence: pd.Series, correct: pd.Series) -> float:
    """Pearson correlation between confidence and correctness.

    With a binary second variable this is the point-biserial correlation. It is
    undefined when either side is constant -- a model that states one
    confidence everywhere, or that is right on every slice of a population.
    """
    if len(confidence) < 3 or confidence.nunique() < 2 or correct.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(confidence.astype(float), correct.astype(float))[0, 1])


def answer_summary(frame: pd.DataFrame) -> dict[str, float]:
    """q, PPV, NPV and the mean confidence on each answer, on one population."""
    says_yes = frame["answer"] == 1
    says_no = frame["answer"] == 0
    return {
        "n": len(frame),
        "q": _mean(says_yes.astype(float)),
        "PPV": _mean(frame.loc[says_yes, "correct"]),
        "NPV": _mean(frame.loc[says_no, "correct"]),
        "mu_yes": _mean(frame.loc[says_yes, "confidence"]),
        "mu_no": _mean(frame.loc[says_no, "confidence"]),
        "mean_confidence": _mean(frame["confidence"]),
    }


def more_confident_answer(summary: dict[str, float]) -> str:
    """Which answer the model states more confidently, "yes" or "no".

    Undefined if the model never gave one of the two answers; NaN comparisons
    are false, so such a model reports the answer it did give.
    """
    mu_yes, mu_no = summary["mu_yes"], summary["mu_no"]
    if np.isnan(mu_yes):
        return "no"
    if np.isnan(mu_no):
        return "yes"
    return "yes" if mu_yes >= mu_no else "no"


def by_stratum(table: pd.DataFrame) -> pd.DataFrame:
    """Table 2: sensitivity, specificity per stratum, balanced accuracy, confidence."""
    rows = []
    for label, model_rows in answered(table).groupby("model_label", sort=False):
        positives = model_rows[model_rows["stratum"] == "BraTS+"]
        brats_negatives = model_rows[model_rows["stratum"] == "BraTS-"]
        ixi = model_rows[model_rows["stratum"] == "IXI"]

        sensitivity = _mean(positives["answer"])
        specificity_brats = 1.0 - _mean(brats_negatives["answer"])
        specificity_ixi = 1.0 - _mean(ixi["answer"])
        by_stratum_conf = [_mean(part["confidence"])
                           for part in (positives, brats_negatives, ixi) if len(part)]

        rows.append({
            "model": label,
            "sensitivity_brats_pos": sensitivity,
            "specificity_brats_neg": specificity_brats,
            "specificity_ixi": specificity_ixi,
            # Balanced accuracy within BraTS: the two classes there come from the
            # same patients and the same scanners, so it is the one summary that
            # does not mix the two sources.
            "balanced_accuracy_brats": (sensitivity + specificity_brats) / 2.0,
            "mean_confidence_min": min(by_stratum_conf),
            "mean_confidence_max": max(by_stratum_conf),
        })
    return pd.DataFrame(rows)


def by_population(table: pd.DataFrame) -> pd.DataFrame:
    """Per model and population: accuracy, rho, and the ingredients of Lemma 1."""
    rows = []
    for label, model_rows in answered(table).groupby("model_label", sort=False):
        for population in ("BraTS", "IXI"):
            part = model_rows[model_rows["population"] == population]
            if part.empty:
                continue
            summary = answer_summary(part)
            rows.append({
                "model": label,
                "population": population,
                "accuracy": _mean(part["correct"]),
                "rho": correlation(part["confidence"], part["correct"]),
                "more_confident_answer": more_confident_answer(summary),
                **summary,
            })
    return pd.DataFrame(rows)


def outcome_confidence(table: pd.DataFrame, population: str | None = None) -> pd.DataFrame:
    """Mean stated confidence on true/false positives and negatives (Figure 3a).

    With `population=None` this pools both populations, which is how the paper
    reports it: the outcome-level means available for four of the models are
    pooled over all labelled slices.
    """
    frame = answered(table)
    if population is not None:
        frame = frame[frame["population"] == population]

    outcome = np.select(
        [(frame["reference"] == 1) & (frame["answer"] == 1),
         (frame["reference"] == 0) & (frame["answer"] == 1),
         (frame["reference"] == 0) & (frame["answer"] == 0),
         (frame["reference"] == 1) & (frame["answer"] == 0)],
        ["true_positive", "false_positive", "true_negative", "false_negative"],
        default="unknown",
    )
    grouped = (frame.assign(outcome=outcome)
                    .groupby(["model_label", "outcome"])["confidence"]
                    .agg(["mean", "size"])
                    .rename(columns={"mean": "mean_confidence", "size": "n"})
                    .reset_index()
                    .rename(columns={"model_label": "model"}))
    return grouped

"""Lemma 1: splitting Cov(c, z) into an answer term and within-answer terms.

A gate can only beat the answer itself if confidence separates right from wrong
*within* an answer. Confidence can correlate with correctness for a duller
reason: the model may simply state one answer more confidently than the other,
and on this population that answer may happen to be right more often.

The law of total covariance, applied to the answer, separates the two:

    Cov(c, z) = q(1-q)(mu1 - mu0)(PPV - NPV)        <- answer term
              + q   * Cov(c, z | yhat = 1)          <- within "yes"
              + (1-q) * Cov(c, z | yhat = 0)        <- within "no"

The answer term depends on the population, not on what the model knows about
its own errors: hold the model fixed, change the case mix, and PPV - NPV can
change sign on its own. That is why the same model's confidence-correctness
correlation points one way on glioma patients and the other way on healthy
volunteers.

Corollary 1(a): where every slice has the same true label, as on IXI, z is
determined by the answer, so both within-answer covariances are zero whatever
the confidence signal is. The sign of rho there records only which answer the
model states more confidently -- it says nothing about self-knowledge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.rates import answer_summary
from src.analysis.study import answered


def _covariance(x: pd.Series, y: pd.Series) -> float:
    """Population covariance (1/n), which is what the identity is written in."""
    if len(x) < 2:
        return 0.0
    return float(np.cov(x.astype(float), y.astype(float), bias=True)[0, 1])


def decompose(frame: pd.DataFrame) -> dict[str, float]:
    """Split Cov(c, z) for one model on one population.

    `residual` is the identity's check: it is zero up to floating point when the
    three terms are computed from the same rows, and a non-zero value means the
    inputs were not consistent.
    """
    rows = answered(frame)
    summary = answer_summary(rows)
    q, mu_yes, mu_no = summary["q"], summary["mu_yes"], summary["mu_no"]
    ppv, npv = summary["PPV"], summary["NPV"]

    total = _covariance(rows["confidence"], rows["correct"])

    # With only one answer given, the answer term is empty and every slice sits
    # in one within-answer group.
    if np.isnan(mu_yes) or np.isnan(mu_no) or np.isnan(ppv) or np.isnan(npv):
        answer_term = 0.0
    else:
        answer_term = q * (1.0 - q) * (mu_yes - mu_no) * (ppv - npv)

    says_yes = rows[rows["answer"] == 1]
    says_no = rows[rows["answer"] == 0]
    within_yes = _covariance(says_yes["confidence"], says_yes["correct"])
    within_no = _covariance(says_no["confidence"], says_no["correct"])
    within_term = q * within_yes + (1.0 - q) * within_no

    return {
        "n": summary["n"],
        "cov_total": total,
        "answer_term": answer_term,
        "within_term": within_term,
        "within_yes": within_yes,
        "within_no": within_no,
        "answer_share": answer_term / total if total else float("nan"),
        "residual": total - (answer_term + within_term),
        "q": q, "PPV": ppv, "NPV": npv, "mu_yes": mu_yes, "mu_no": mu_no,
    }


def decomposition_table(table: pd.DataFrame,
                        populations: tuple[str, ...] = ("BraTS", "IXI")) -> pd.DataFrame:
    rows = []
    for label, model_rows in table.groupby("model_label", sort=False):
        for population in populations:
            part = model_rows[model_rows["population"] == population]
            if part.empty:
                continue
            rows.append({"model": label, "population": population, **decompose(part)})
    return pd.DataFrame(rows)


def answer_term_sign_flips(table: pd.DataFrame) -> pd.DataFrame:
    """Does the answer term change sign between the two populations, and why?

    The term is a product of three factors. q(1-q) is positive whenever a model
    gives both answers, so the sign is carried by (mu1 - mu0), which answer the
    model prefers, and (PPV - NPV), which answer is actually more reliable on
    this population. A model keeps its preference across populations while
    PPV - NPV follows the case mix, and that is the sign flip.
    """
    decomposed = decomposition_table(table)
    wide = decomposed.pivot(index="model", columns="population")

    rows = []
    for model in wide.index:
        record: dict[str, object] = {"model": model}
        for population in ("BraTS", "IXI"):
            record[f"answer_term_{population}"] = wide[("answer_term", population)][model]
            record[f"mu_gap_{population}"] = (wide[("mu_yes", population)][model]
                                              - wide[("mu_no", population)][model])
            record[f"pv_gap_{population}"] = (wide[("PPV", population)][model]
                                              - wide[("NPV", population)][model])
        record["flips_sign"] = bool(
            np.sign(record["answer_term_BraTS"]) * np.sign(record["answer_term_IXI"]) < 0)
        rows.append(record)
    return pd.DataFrame(rows)

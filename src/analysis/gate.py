"""The deferral gate: coverage, selective risk, and the largest low-error set.

A gate with threshold tau accepts the answers with confidence at least tau and
defers the rest. On n slices (Eq. 1),

    coverage(tau) = (1/n) * #{i : c_i >= tau}
    risk(tau)     = #{i : c_i >= tau, answer wrong} / #{i : c_i >= tau}

and C_alpha (Eq. 2) is the largest coverage any threshold reaches while keeping
risk at or below alpha, or 0 if none does.

Three things to keep in mind when reading what this module returns.

*n counts deferred slices.* A slice the model did not answer, or answered
without a parseable confidence, can never be accepted, but it is still part of
the task. Coverage therefore says how much of the slice task could be
automated, not how much of the answered part could be.

*A threshold cannot split a tie.* Most models put hundreds of slices on a
single value such as 0.90, so the candidate thresholds are exactly the distinct
confidence values a model used, and an operating point takes a whole tie or
none of it.

*C_alpha is chosen in sample.* The threshold is picked after seeing the
outcomes, so a zero is strong evidence that nothing works, while a positive
value is the best coverage observed here and not a guarantee for new data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANSWER_ONLY_RULES = ("all_yes", "all_no", "all_answers")


def operating_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Every operating point a threshold can reach, from the highest down.

    Rows are cumulative: each adds one whole confidence level to the accepted
    set. `n` is the size of the slice set, deferred slices included.
    """
    n = len(frame)
    answered = frame[frame["answered"]].sort_values("confidence", ascending=False,
                                                    kind="mergesort")
    if n == 0 or answered.empty:
        return pd.DataFrame(columns=["threshold", "accepted", "errors", "coverage", "risk"])

    levels = (answered.groupby("confidence", sort=False)
                      .agg(accepted=("correct", "size"), correct=("correct", "sum")))
    levels = levels.sort_index(ascending=False)

    accepted = levels["accepted"].cumsum()
    errors = (levels["accepted"] - levels["correct"]).cumsum()
    return pd.DataFrame({
        "threshold": levels.index.to_numpy(float),
        "accepted": accepted.to_numpy(int),
        "errors": errors.to_numpy(int),
        "coverage": accepted.to_numpy(float) / n,
        "risk": errors.to_numpy(float) / accepted.to_numpy(float),
    })


def max_coverage_at_risk(frame: pd.DataFrame, alpha: float) -> dict[str, float]:
    """C_alpha and the operating point that attains it."""
    points = operating_points(frame)
    feasible = points[points["risk"] <= alpha + 1e-12]
    if feasible.empty:
        return {"alpha": alpha, "coverage": 0.0, "accepted": 0,
                "threshold": float("nan"), "risk": float("nan")}

    best = feasible.loc[feasible["coverage"].idxmax()]
    return {"alpha": alpha, "coverage": float(best["coverage"]),
            "accepted": int(best["accepted"]), "threshold": float(best["threshold"]),
            "risk": float(best["risk"])}


def answer_only_rule(frame: pd.DataFrame, alpha: float) -> dict[str, object]:
    """The best rule that ignores confidence: accept all "yes", all "no", or all.

    This is the baseline a gate has to beat. If a model's confidence carries no
    information beyond which answer it gave, a gate can only ever accept a whole
    answer class, and then it cannot do better than this.
    """
    n = len(frame)
    answered = frame[frame["answered"]]
    masks = {
        "all_yes": answered["answer"] == 1,
        "all_no": answered["answer"] == 0,
        "all_answers": pd.Series(True, index=answered.index),
    }

    best = {"rule": "none", "coverage": 0.0, "accepted": 0, "risk": float("nan")}
    for rule, mask in masks.items():
        subset = answered[mask]
        if subset.empty:
            continue
        risk = 1.0 - float(subset["correct"].mean())
        coverage = len(subset) / n if n else 0.0
        if risk <= alpha + 1e-12 and coverage > best["coverage"]:
            best = {"rule": rule, "coverage": coverage, "accepted": len(subset), "risk": risk}
    return best


def gate_table(table: pd.DataFrame, alphas: list[float],
               population: str | None = "BraTS") -> pd.DataFrame:
    """Table 3: accuracy and C_alpha per model, on one population."""
    frame = table if population is None else table[table["population"] == population]

    rows = []
    for label, model_rows in frame.groupby("model_label", sort=False):
        answered = model_rows[model_rows["answered"]]
        row = {
            "model": label,
            "slices": len(model_rows),
            "answered": len(answered),
            "accuracy": float(answered["correct"].mean()) if len(answered) else float("nan"),
        }
        for alpha in alphas:
            result = max_coverage_at_risk(model_rows, alpha)
            row[f"C_{alpha:.2f}"] = result["coverage"]
            row[f"n_{alpha:.2f}"] = result["accepted"]
            row[f"tau_{alpha:.2f}"] = result["threshold"]
        rows.append(row)
    return pd.DataFrame(rows)


def gate_vs_answer_only(table: pd.DataFrame, alpha: float,
                        population: str | None = "BraTS") -> pd.DataFrame:
    """Table 4: the gate against the rule that ignores confidence."""
    from src.analysis.rates import answer_summary, more_confident_answer
    from src.analysis.study import answered as answered_rows

    frame = table if population is None else table[table["population"] == population]

    rows = []
    for label, model_rows in frame.groupby("model_label", sort=False):
        summary = answer_summary(answered_rows(model_rows))
        favoured = more_confident_answer(summary)
        said = model_rows["answer"] == (1 if favoured == "yes" else 0)

        baseline = answer_only_rule(model_rows, alpha)
        rows.append({
            "model": label,
            "more_confident_answer": favoured,
            "share_right_given_that_answer": summary["PPV"] if favoured == "yes" else summary["NPV"],
            "share_of_slices_given_that_answer": float(said.mean()),
            "gate_coverage": max_coverage_at_risk(model_rows, alpha)["coverage"],
            "answer_only_coverage": baseline["coverage"],
            "answer_only_rule": baseline["rule"],
        })
    return pd.DataFrame(rows)


def errors_in_accepted_set(frame: pd.DataFrame, alpha: float) -> dict[str, int]:
    """What the accepted set at C_alpha gets wrong, split by error type."""
    result = max_coverage_at_risk(frame, alpha)
    if not result["accepted"]:
        return {"accepted": 0, "false_positives": 0, "false_negatives": 0}

    accepted = frame[frame["answered"] & (frame["confidence"] >= result["threshold"])]
    wrong = accepted[accepted["correct"] == 0]
    return {
        "accepted": len(accepted),
        "false_positives": int((wrong["answer"] == 1).sum()),
        "false_negatives": int((wrong["answer"] == 0).sum()),
    }


def specificity_needed(sensitivity: float, pi: float, alpha: float) -> float:
    """Specificity at which accepting every "yes" answer would hit error `alpha`.

    From the definition of PPV at case mix pi, solving PPV = 1 - alpha gives

        specificity = 1 - [alpha / (1 - alpha)] * [pi / (1 - pi)] * sensitivity

    which is the "specificity needed" column of Table 5. A value above 1 means
    no specificity would do it at that sensitivity.
    """
    if not 0.0 < pi < 1.0 or np.isnan(sensitivity):
        return float("nan")
    return 1.0 - (alpha / (1.0 - alpha)) * (pi / (1.0 - pi)) * sensitivity

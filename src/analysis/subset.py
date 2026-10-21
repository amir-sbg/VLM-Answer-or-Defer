"""The large-lesion, distant-negative subset of BraTS (Section 4.6, Table 5).

The strict result could in principle be an artifact of a debatable reference: a
compartment delineated on the multimodal study need not be visible on the one
sequence shown, and a mask-negative slice from a treated patient may still show
change outside the annotation. This subset is a partial check on that. It keeps
positives whose annotated lesion is large on a sequence where it should be
visible, and negatives far from any annotation, then reweights the two to the
natural case mix so the restriction does not quietly change the prior.

It restricts the images. It does not re-read them, so it cannot rule out a
reference error -- only show that the result does not depend on the hardest
cases. A blinded expert read of the displayed slices is the test that would.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.gate import answer_only_rule, max_coverage_at_risk, specificity_needed
from src.analysis.study import answered


def select(table: pd.DataFrame, subset_cfg: dict) -> pd.DataFrame:
    """Keep large-lesion positives and distant negatives, within BraTS."""
    sequences = set(subset_cfg["lesion_sequences"])
    min_area = int(subset_cfg["min_lesion_px"])
    min_distance = float(subset_cfg["min_negative_distance_mm"])
    restrict_negatives = bool(subset_cfg.get("restrict_negatives_to_sequences", True))

    brats = table[table["population"] == "BraTS"]
    on_sequence = brats["modality"].isin(sequences)

    keep_positive = (brats["stratum"] == "BraTS+") & on_sequence & \
                    (brats["lesion_area_px"] >= min_area)
    keep_negative = (brats["stratum"] == "BraTS-") & \
                    (brats["mask_distance_mm"] > min_distance)
    if restrict_negatives:
        keep_negative &= on_sequence

    return brats[keep_positive | keep_negative].copy()


def reweighted_rates(frame: pd.DataFrame, pi: float) -> dict[str, float]:
    """Sensitivity and specificity on the subset, and PPV/NPV at case mix `pi`.

    Restricting to large lesions raises the share of positives; reporting the
    predictive values at the natural mix instead of the subset's own mix keeps
    them comparable with the full-BraTS numbers.
    """
    rows = answered(frame)
    positives = rows[rows["reference"] == 1]
    negatives = rows[rows["reference"] == 0]

    sensitivity = float(positives["answer"].mean()) if len(positives) else float("nan")
    specificity = 1.0 - float(negatives["answer"].mean()) if len(negatives) else float("nan")

    true_pos, false_pos = pi * sensitivity, (1 - pi) * (1 - specificity)
    true_neg, false_neg = (1 - pi) * specificity, pi * (1 - sensitivity)
    return {
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv_at_pi": true_pos / (true_pos + false_pos) if true_pos + false_pos else float("nan"),
        "npv_at_pi": true_neg / (true_neg + false_neg) if true_neg + false_neg else float("nan"),
    }


def answer_only_at_pi(sensitivity: float, specificity: float, pi: float,
                      alpha: float) -> dict[str, float]:
    """Best answer-only rule after reweighting the subset to case mix `pi`.

    The subset is built by restricting the images, which raises the share of
    positives well above the natural mix. Reporting the answer-only rule on the
    subset as it stands would therefore flatter it, so the shares and the
    predictive values are both taken at `pi`: a "yes" answer is given on
    pi*sens + (1-pi)(1-spec) of slices and is right PPV(pi) of the time.
    """
    yes_share = pi * sensitivity + (1 - pi) * (1 - specificity)
    no_share = (1 - pi) * specificity + pi * (1 - sensitivity)
    true_pos, true_neg = pi * sensitivity, (1 - pi) * specificity

    best = {"coverage": 0.0, "rule": "none", "accuracy": float("nan")}
    for rule, share, right in (("all_yes", yes_share, true_pos),
                               ("all_no", no_share, true_neg)):
        if share <= 0:
            continue
        accuracy = right / share
        if accuracy >= 1 - alpha - 1e-12 and share > best["coverage"]:
            best = {"coverage": float(share), "rule": rule, "accuracy": float(accuracy)}
    return best


def subset_table(table: pd.DataFrame, subset_cfg: dict, alphas: list[float],
                 models: list[str] | None = None) -> pd.DataFrame:
    """Table 5: the gate on the subset, against C_alpha on all of BraTS."""
    pi = float(subset_cfg["reweight_to_pi"])
    alpha_strict = min(alphas)
    alpha_loose = max(alphas)

    selected = select(table, subset_cfg)
    full = table[table["population"] == "BraTS"]
    labels = models or sorted(selected["model_label"].unique())

    rows = []
    for label in labels:
        on_subset = selected[selected["model_label"] == label]
        on_full = full[full["model_label"] == label]
        if on_subset.empty:
            continue

        rates = reweighted_rates(on_subset, pi)
        loose = answer_only_at_pi(rates["sensitivity"], rates["specificity"], pi, alpha_loose)

        record = {
            "model": label,
            **{k: rates[k] for k in ("n_positive", "n_negative", "sensitivity", "specificity")},
            "specificity_needed_for_strict_target":
                specificity_needed(rates["sensitivity"], pi, alpha_strict),
            f"answer_only_{alpha_loose:.2f}_at_pi": loose["coverage"],
            "answer_only_rule_at_pi": loose["rule"],
        }
        for alpha in alphas:
            record[f"C_{alpha:.2f}_all_brats"] = max_coverage_at_risk(on_full, alpha)["coverage"]
            record[f"C_{alpha:.2f}_subset"] = max_coverage_at_risk(on_subset, alpha)["coverage"]
            record[f"answer_only_{alpha:.2f}_observed"] = \
                answer_only_rule(on_subset, alpha)["coverage"]
        rows.append(record)
    return pd.DataFrame(rows)

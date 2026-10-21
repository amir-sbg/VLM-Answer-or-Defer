"""Sensitivity and stated confidence by lesion size (Section 4.5, Figure 5).

If stated confidence tracked how hard a slice is, it would fall where the
lesion is small and sensitivity collapses. Binning the positive slices by the
in-plane area of the annotated lesion tests that directly.

Two caveats travel with the numbers. These are bin means, so they do not rule
out confidence ranking slices correctly *within* a bin. And every slice here is
reference-positive, so part of any change in mean confidence across bins comes
from the shift in the mix of "yes" and "no" answers rather than from the model
becoming more or less sure of a fixed answer -- which is why the per-answer
means are reported alongside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.study import answered


def bin_edges(lesion_cfg: dict) -> tuple[list[int], list[str]]:
    """Bin boundaries and their labels, from the configured edges."""
    edges = list(lesion_cfg["bins"])
    labels = [f"{low:,}-{high:,}" for low, high in zip(edges, edges[1:])]
    if lesion_cfg.get("top_open", True):
        labels.append(f">{edges[-1]:,}")
        edges = edges + [np.inf]
    return edges, labels


def assign_bins(frame: pd.DataFrame, lesion_cfg: dict) -> pd.Series:
    edges, labels = bin_edges(lesion_cfg)
    return pd.cut(frame["lesion_area_px"], bins=edges, labels=labels,
                  right=False, include_lowest=True)


def sensitivity_by_area(table: pd.DataFrame, lesion_cfg: dict) -> pd.DataFrame:
    """Per model and area bin: sensitivity, mean confidence, and the answer mix."""
    positives = answered(table[table["stratum"] == "BraTS+"]).copy()
    positives["area_bin"] = assign_bins(positives, lesion_cfg)

    grouped = (positives.groupby(["model_label", "area_bin"], observed=True)
               .apply(lambda part: pd.Series({
                   "n": len(part),
                   "sensitivity": float(part["answer"].mean()),
                   "mean_confidence": float(part["confidence"].mean()),
                   "mean_confidence_yes": float(part.loc[part["answer"] == 1, "confidence"].mean()),
                   "mean_confidence_no": float(part.loc[part["answer"] == 0, "confidence"].mean()),
               }), include_groups=False)
               .reset_index()
               .rename(columns={"model_label": "model"}))
    grouped["n"] = grouped["n"].astype(int)
    return grouped


def span_across_bins(by_area: pd.DataFrame) -> pd.DataFrame:
    """How far sensitivity and mean confidence move from the first bin to the last."""
    rows = []
    for model, part in by_area.groupby("model", sort=False):
        part = part.sort_values("area_bin", key=lambda s: s.map(
            {b: i for i, b in enumerate(by_area["area_bin"].unique())}))
        first, last = part.iloc[0], part.iloc[-1]
        rows.append({
            "model": model,
            "sensitivity_smallest_bin": first["sensitivity"],
            "sensitivity_largest_bin": last["sensitivity"],
            "sensitivity_change": last["sensitivity"] - first["sensitivity"],
            "confidence_smallest_bin": first["mean_confidence"],
            "confidence_largest_bin": last["mean_confidence"],
            "confidence_change": last["mean_confidence"] - first["mean_confidence"],
        })
    return pd.DataFrame(rows)

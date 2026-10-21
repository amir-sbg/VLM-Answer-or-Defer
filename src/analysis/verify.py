"""Check every number the paper prints against the same number recomputed here.

A paper that promises to release its per-slice outputs is making each of its
quantitative claims falsifiable from those outputs. This module does the
falsifying first: it reads the values as printed (docs/reported_values.yaml),
recomputes each one from the released decisions, and writes a row per
comparison saying whether they agree to the paper's own printing precision.

It exits non-zero when something does not match, so a revision that changes the
pipeline cannot quietly leave a stale number in the manuscript.

Usage:
    python -m src.analysis.verify --output results/verification.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import casemix, gate, lesion, rates, study, subset
from src.utils import get_logger, load_yaml, setup_logging, write_csv

log = get_logger(__name__)


def compare(quantity: str, model: str, reported: float, recomputed: float,
            tolerance: float) -> dict:
    difference = (float("nan") if reported is None or np.isnan(recomputed)
                  else abs(recomputed - reported))
    return {
        "quantity": quantity,
        "model": model,
        "reported": reported,
        "recomputed": None if np.isnan(recomputed) else round(float(recomputed), 4),
        "difference": None if np.isnan(difference) else round(difference, 4),
        "matches": bool(difference <= tolerance) if difference == difference else False,
    }


def compare_block(name: str, block: dict, recomputed: pd.DataFrame,
                  index_column: str = "model") -> list[dict]:
    """Compare one table block, keyed by model and column name."""
    tolerance = float(block.get("tolerance", 0.0))
    indexed = recomputed.set_index(index_column)

    rows = []
    for model, reported_values in block["rows"].items():
        for column, reported in zip(block["columns"], reported_values):
            if model not in indexed.index or column not in indexed.columns:
                value = float("nan")
            else:
                value = float(indexed.loc[model, column])
            rows.append(compare(f"{name}.{column}", model, reported, value, tolerance))
    return rows


def verify(table: pd.DataFrame, cfg: dict, reported: dict) -> pd.DataFrame:
    alphas = cfg["gate"]["alphas"]
    models = study.ordered_models(table)
    rows: list[dict] = []

    rows += compare_block("table2_rates", reported["table2_rates"], rates.by_stratum(table))

    gate_table = gate.gate_table(table, alphas, population="BraTS")
    rows += compare_block("table3_gate", reported["table3_gate"], gate_table)

    answer_only = gate.gate_vs_answer_only(table, max(alphas), population="BraTS")
    rows += compare_block("table4_answer_only", reported["table4_answer_only"], answer_only)
    favoured = answer_only.set_index("model")["more_confident_answer"]
    for model, stated in reported["table4_answer_only"]["more_confident_answer"].items():
        actual = favoured.get(model)
        rows.append({"quantity": "table4_answer_only.more_confident_answer", "model": model,
                     "reported": stated, "recomputed": actual, "difference": None,
                     "matches": bool(actual == stated)})

    subset_table = subset.subset_table(table, cfg["subset"], alphas, models)
    rows += compare_block("table5_subset", reported["table5_subset"], subset_table)
    sizes = reported["table5_subset"]["subset_sizes"]
    selected = subset.select(table, cfg["subset"])
    one_model = selected[selected["model_label"] == models[0]]
    for stratum, key in (("BraTS+", "positive"), ("BraTS-", "negative")):
        rows.append(compare("table5_subset.n_slices", stratum, sizes[key],
                            float((one_model["stratum"] == stratum).sum()), 0))

    outcomes = (rates.outcome_confidence(table)
                .pivot(index="model", columns="outcome", values="mean_confidence")
                .reset_index())
    rows += compare_block("outcome_confidence", reported["outcome_confidence"], outcomes)

    rho = (rates.by_population(table).pivot(index="model", columns="population", values="rho")
           .reset_index())
    rows += compare_block("rho_by_population", reported["rho_by_population"], rho)

    strata = study.summarize_strata(table).set_index("stratum")["slices"]
    for stratum, count in reported["strata"]["slices"].items():
        rows.append(compare("strata.slices", stratum, count,
                            float(strata.get(stratum, np.nan)), 0))
    rows.append(compare("strata.total_slices", "all", reported["strata"]["total_slices"],
                        float(strata.sum()), 0))

    rows += verify_claims(table, cfg, reported["claims"], gate_table, subset_table, rho)
    return pd.DataFrame(rows)


def verify_claims(table: pd.DataFrame, cfg: dict, claims: dict, gate_table: pd.DataFrame,
                  subset_table: pd.DataFrame, rho: pd.DataFrame) -> list[dict]:
    """The prose claims, which are the ones a reader is most likely to quote."""
    strict = f"C_{min(cfg['gate']['alphas']):.2f}"

    def claim(name: str, holds: bool, detail: str) -> dict:
        return {"quantity": f"claims.{name}", "model": detail,
                "reported": claims[name], "recomputed": holds, "difference": None,
                "matches": bool(holds == claims[name])}

    failing = gate_table.loc[gate_table[strict] > 0, "model"].tolist()
    rows = [claim("no_model_meets_strict_target_on_brats", not failing,
                  "all models" if not failing else "nonzero: " + ", ".join(failing))]

    failing_subset = subset_table.loc[subset_table[f"{strict}_subset"] > 0, "model"].tolist()
    rows.append(claim("no_model_meets_strict_target_on_subset", not failing_subset,
                      "all models" if not failing_subset
                      else "nonzero: " + ", ".join(failing_subset)))

    signs = rho.set_index("model")
    same_sign = [m for m, row in signs.iterrows()
                 if np.sign(row["BraTS"]) == np.sign(row["IXI"])]
    rows.append(claim("rho_flips_sign_for_every_model", not same_sign,
                      "all models" if not same_sign else "same sign: " + ", ".join(same_sign)))

    by_area = lesion.sensitivity_by_area(table, cfg["lesion_area"])
    smallest = by_area[by_area["area_bin"] == by_area["area_bin"].iloc[0]]["sensitivity"]
    reported_range = claims["smallest_lesion_bin_sensitivity_range"]
    tolerance = 0.01
    rows.append({
        "quantity": "claims.smallest_lesion_bin_sensitivity_range", "model": "range",
        "reported": f"{reported_range[0]:.2f}-{reported_range[1]:.2f}",
        "recomputed": f"{smallest.min():.2f}-{smallest.max():.2f}",
        "difference": round(max(abs(smallest.min() - reported_range[0]),
                                abs(smallest.max() - reported_range[1])), 4),
        "matches": bool(abs(smallest.min() - reported_range[0]) <= tolerance
                        and abs(smallest.max() - reported_range[1]) <= tolerance),
    })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", default="results/tumor_decisions.csv.gz")
    ap.add_argument("--study", default="config/study.yaml")
    ap.add_argument("--reported", default="docs/reported_values.yaml")
    ap.add_argument("--output", default="results/verification.csv")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="write the report and exit 0 even where values disagree")
    args = ap.parse_args()

    setup_logging()
    report = verify(study.load(args.table), load_yaml(args.study), load_yaml(args.reported))
    write_csv(report, args.output)

    matched = int(report["matches"].sum())
    log.info(f"{matched}/{len(report)} reported values reproduce")
    mismatches = report[~report["matches"]]
    if not mismatches.empty:
        log.warning("values that do not reproduce:\n" + mismatches.to_string(index=False))
    log.info(f"report -> {args.output}")

    if not mismatches.empty and not args.allow_mismatch:
        sys.exit(1)


if __name__ == "__main__":
    main()

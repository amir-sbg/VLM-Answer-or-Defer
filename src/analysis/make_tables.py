"""Build every table in the paper from the study table, as CSV and as LaTeX.

Written files (under --output-dir):
    table1_strata.csv              the three strata
    table2_rates.csv               the tumor call by stratum
    table2_intervals.csv           subject-clustered intervals for those rates
    table3_gate.csv                stated confidence as a deferral gate on BraTS
    table3_gate_ixi.csv            the same gate on healthy volunteers
    table4_answer_only.csv         the gate against a rule that ignores confidence
    table5_subset.csv              the large-lesion, distant-negative subset
    rho_by_population.csv          rho and the ingredients of Lemma 1
    outcome_confidence.csv         mean confidence on TP / FP / TN / FN
    decomposition.csv              the answer term against the within-answer terms
    case_mix.csv                   error of the favoured answer against pi
    lesion_area.csv                sensitivity and confidence by lesion area
    robustness_positivity.csv      the same rates at other positivity thresholds
    accepted_set_errors.csv        what the accepted set gets wrong at each budget

Usage:
    python -m src.analysis.make_tables --output-dir results/tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis import bootstrap, casemix, decomposition, gate, lesion, rates, study, subset
from src.utils import get_logger, load_yaml, setup_logging, write_csv

log = get_logger(__name__)

# Tables worth having as LaTeX, with the caption-level rounding the paper uses.
LATEX_TABLES = {
    "table2_rates": 3,
    "table3_gate": 3,
    "table4_answer_only": 3,
    "table5_subset": 2,
}


def order_rows(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Put the models in the order of the paper's tables."""
    if "model" not in frame.columns:
        return frame
    position = {label: i for i, label in enumerate(models)}
    # Only the model order is imposed. A stable sort keeps whatever order the
    # analysis produced within a model, which matters for the area bins: sorting
    # their labels would put "100-400" before "25-100".
    return (frame.assign(_order=frame["model"].map(position))
                 .sort_values("_order", kind="mergesort")
                 .drop(columns="_order")
                 .reset_index(drop=True))


def robustness_over_thresholds(manifest_path: str, responses_dir: str, tasks_path: str,
                               models_path: str, study_cfg: dict,
                               thresholds: list[int]) -> pd.DataFrame:
    """Re-derive the strata at each positivity threshold and recompute the rates.

    This is the check of Section 4.6: does the reference threshold drive the
    result? It re-runs the join, because a different threshold moves slices
    between BraTS+ and BraTS-.
    """
    from src.utils import model_labels

    manifest = pd.read_csv(manifest_path)
    labels = model_labels(models_path)
    records = study.read_transcripts(responses_dir, study_cfg["decision"], list(labels))
    tasks_cfg = load_yaml(tasks_path)

    frames = []
    for threshold in thresholds:
        reference_cfg = {**study_cfg["reference"], "positivity_threshold_px": threshold}
        table = study.build(manifest, records, tasks_cfg, reference_cfg, labels)
        frames.append(rates.by_stratum(table).assign(positivity_threshold_px=threshold))
    return pd.concat(frames, ignore_index=True)


def accepted_set_errors(table: pd.DataFrame, alphas: list[float]) -> pd.DataFrame:
    """Which errors survive in the accepted set at each budget (Section 4.3)."""
    brats = table[table["population"] == "BraTS"]
    rows = []
    for label, model_rows in brats.groupby("model_label", sort=False):
        for alpha in alphas:
            rows.append({"model": label, "alpha": alpha,
                         **gate.errors_in_accepted_set(model_rows, alpha)})
    return pd.DataFrame(rows)


def write_latex(frame: pd.DataFrame, path: Path, decimals: int) -> None:
    numeric = frame.select_dtypes("number").columns
    frame.round({c: decimals for c in numeric}).to_latex(path, index=False, escape=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", default="results/tumor_decisions.csv.gz")
    ap.add_argument("--manifest", default="data/exam_set.csv")
    ap.add_argument("--responses-dir", default="raw_responses")
    ap.add_argument("--tasks", default="config/tasks.yaml")
    ap.add_argument("--models", default="config/models.yaml")
    ap.add_argument("--study", default="config/study.yaml")
    ap.add_argument("--output-dir", default="results/tables")
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="skip the interval table, which is the slow part")
    ap.add_argument("--no-robustness", action="store_true",
                    help="skip the positivity-threshold sweep, which re-reads the transcripts")
    args = ap.parse_args()

    setup_logging()
    cfg = load_yaml(args.study)
    alphas = cfg["gate"]["alphas"]
    table = study.load(args.table)
    models = study.ordered_models(table)
    out = Path(args.output_dir)

    built: dict[str, pd.DataFrame] = {
        "table1_strata": study.summarize_strata(table),
        "table2_rates": rates.by_stratum(table),
        "table3_gate": gate.gate_table(table, alphas, population="BraTS"),
        "table3_gate_ixi": gate.gate_table(table, alphas, population="IXI"),
        "table4_answer_only": gate.gate_vs_answer_only(table, max(alphas), population="BraTS"),
        "table5_subset": subset.subset_table(table, cfg["subset"], alphas, models),
        "rho_by_population": rates.by_population(table),
        "outcome_confidence": rates.outcome_confidence(table),
        "decomposition": decomposition.decomposition_table(table),
        "answer_term_signs": decomposition.answer_term_sign_flips(table),
        "case_mix": casemix.summary_table(table, cfg["case_mix"]),
        "case_mix_curves": casemix.error_curves(table, cfg["case_mix"]),
        "lesion_area": lesion.sensitivity_by_area(table, cfg["lesion_area"]),
        "accepted_set_errors": accepted_set_errors(table, alphas),
    }

    if not args.no_bootstrap:
        built["table2_intervals"] = bootstrap.rate_intervals(table, cfg["bootstrap"])
    if not args.no_robustness:
        built["robustness_positivity"] = robustness_over_thresholds(
            args.manifest, args.responses_dir, args.tasks, args.models, cfg,
            cfg["robustness"]["positivity_thresholds_px"])

    for name, frame in built.items():
        frame = order_rows(frame, models)
        write_csv(frame, out / f"{name}.csv")
        if name in LATEX_TABLES:
            write_latex(frame, out / f"{name}.tex", LATEX_TABLES[name])
        log.info(f"{name}: {len(frame)} rows")

    log.info(f"{len(built)} tables -> {out}")


if __name__ == "__main__":
    main()

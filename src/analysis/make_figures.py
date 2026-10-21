"""Render the paper's data figures from the study table.

Figures 3, 4 and 5 are built from the study table alone. Figure 2 additionally
needs the slice PNGs, the BraTS segmentations and the transcripts, since it
shows a real exchange with its reference outline; it is skipped with a warning
if those are not present, so the rest still builds from the committed table.

Figure 1 is a schematic of the study design with no data behind it and is drawn
in the paper source, not here.

Usage:
    python -m src.analysis.make_figures --output-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis import casemix, lesion, rates, study
from src.figures import casemix as fig_casemix
from src.figures import confidence as fig_confidence
from src.figures import lesion as fig_lesion
from src.utils import get_logger, load_yaml, project_path, read_jsonl, setup_logging

log = get_logger(__name__)

# The four models for which the paper reports outcome-level confidences.
OUTCOME_MODELS = ["Gemma-4-12B", "InternVL2.5-8B", "GLM-4.6V-Flash", "Qwen3.6-27B"]


def build_example_figure(table: pd.DataFrame, cfg: dict, model: str, manifest_path: str,
                         output_dir: Path) -> Path | None:
    """Figure 2, if the images and transcripts this repository does not ship are present."""
    from src.figures import examples
    from src.prompts.render import render_item
    from src.scoring.parse import is_reasoning_model
    from src.utils import model_labels

    provenance_path = project_path("data/slice_provenance.jsonl")
    if not provenance_path.exists() or not project_path(manifest_path).exists():
        log.warning("the slices and their provenance are absent; skipping Figure 2")
        return None

    # The study table carries no image path, since it is meant to be readable
    # without the images; the manifest supplies it here.
    paths = pd.read_csv(project_path(manifest_path), usecols=["image_id", "path"])
    panels = examples.choose_panels(table.merge(paths, on="image_id", how="left"), model)
    if panels.empty:
        log.warning(f"no panels qualify for {model}; skipping Figure 2")
        return None

    provenance = {rec["image_id"]: rec for rec in read_jsonl(provenance_path)}
    decision = cfg["decision"]
    model_id = next((mid for mid, label in model_labels().items() if label == model), None)
    transcripts = {rec["image_id"]: rec for rec in
                   study.read_transcripts("raw_responses", decision, [model_id])}

    first = panels.iloc[0]
    record = transcripts.get(first["image_id"])
    if record is None:
        log.warning("no transcript found for the first panel; skipping Figure 2")
        return None

    tasks_cfg = load_yaml("config/tasks.yaml")
    prompts_cfg = load_yaml("config/prompts.yaml")
    rendered = render_item(
        pd.Series({"image_id": first["image_id"], "label_tumor_present": "yes"}),
        decision["task"], tasks_cfg["tasks"][decision["task"]], decision["phrasing"],
        prompts_cfg["phrasings"][decision["phrasing"]], prompts_cfg["system_prompt"])
    prompt_text = next(r["prompt_text"] for r in rendered if r["format"] == decision["format"])

    response = record.get("response", "")
    if is_reasoning_model(record.get("model", "")) and "</think>" in response:
        trace, answer = response.rsplit("</think>", 1)
        words = len(trace.split())
        shown = f"[thinking trace, ~{words} words, omitted] </think>\n{answer.strip()}"
    else:
        shown = response.strip()

    parsed = (f"Answer = {'yes' if first['answer'] == 1 else 'no'}, "
              f"stated confidence c = {first['confidence']:.2f}")
    return examples.make(panels, provenance, prompt_text, shown,
                         f"{model}\n{parsed}", output_dir / "fig2_example.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", default="results/tumor_decisions.csv.gz")
    ap.add_argument("--study", default="config/study.yaml")
    ap.add_argument("--manifest", default="data/exam_set.csv")
    ap.add_argument("--output-dir", default="figures")
    ap.add_argument("--example-model", default="Qwen3.6-27B")
    ap.add_argument("--skip-example", action="store_true")
    args = ap.parse_args()

    setup_logging()
    cfg = load_yaml(args.study)
    table = study.load(args.table)
    order = study.ordered_models(table)
    out = Path(args.output_dir)

    written = []
    if not args.skip_example:
        path = build_example_figure(table, cfg, args.example_model, args.manifest, out)
        if path:
            written.append(path)

    outcome_models = [m for m in OUTCOME_MODELS if m in order]
    written.append(fig_confidence.make(
        outcome_confidence=rates.outcome_confidence(table),
        rho=rates.by_population(table)[["model", "population", "rho"]],
        order=order, outcome_models=outcome_models,
        path=out / "fig3_confidence.pdf"))

    natural_pi = casemix.measured_rates(table[table["model_label"] == order[0]])["natural_pi"]
    written.append(fig_casemix.make(
        curves=casemix.error_curves(table, cfg["case_mix"]),
        natural_pi=natural_pi, alphas=cfg["gate"]["alphas"], order=order,
        path=out / "fig4_case_mix.pdf"))

    written.append(fig_lesion.make(
        by_area=lesion.sensitivity_by_area(table, cfg["lesion_area"]),
        order=order, path=out / "fig5_lesion_area.pdf"))

    for path in written:
        log.info(f"wrote {path} (+ .png)")


if __name__ == "__main__":
    main()

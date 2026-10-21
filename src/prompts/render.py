"""Render the prompt matrix: manifest x tasks x phrasings x {MC, OE}.

One rule matters for the deferral analysis. Multiple-choice option order is
shuffled per item from a seed derived from the item itself, and both the
shuffled list and the original indices are recorded. A model that always picks
"B" then answers at chance rather than scoring whatever happens to sit there,
and grading can still map the chosen letter back to the label it stood for.
"""
from __future__ import annotations

import hashlib
import string
from typing import TypedDict

import pandas as pd

from src.utils import get_logger, shuffle_options

log = get_logger(__name__)

LETTERS = list(string.ascii_uppercase)

# Which manifest column holds the reference for each task. T3-ISBRAIN has no
# column: every slice in this study is a brain MRI.
LABEL_COLUMN = {
    "T1-MOD": "label_modality",
    "T2-PLANE": "label_plane",
    "T3-ISBRAIN": None,
    "T4-TUMOR": "label_tumor_present",
    "T5-LAT": "label_laterality",
}


class PromptRow(TypedDict):
    prompt_id: str
    image_id: str
    task: str
    phrasing: str
    format: str
    prompt_text: str
    system_prompt: str
    mc_options: list[str]
    mc_option_order: list[int]
    mc_correct_letter: str


def item_seed(image_id: str, task: str, phrasing: str) -> int:
    """A stable per-item seed. sha256, not hash(): hash() is salted per process."""
    digest = hashlib.sha256(f"{image_id}|{task}|{phrasing}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def format_options(options: list[str]) -> str:
    return "  ".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))


def reference_label(row: pd.Series, task: str) -> str:
    if task == "T3-ISBRAIN":
        return "yes"
    column = LABEL_COLUMN.get(task)
    return str(row.get(column, "")).strip() if column else ""


def correct_letter(options: list[str], label: str, aliases: dict[str, list[str]]) -> str:
    """Which option letter the reference label corresponds to, or '?' if none."""
    target = label.lower().strip()
    if not target:
        return "?"
    for i, option in enumerate(options):
        candidates = {option.lower().strip()}
        candidates.update(a.lower() for a in aliases.get(option, []))
        if target in candidates:
            return LETTERS[i]
    return "?"


def task_applies(task_cfg: dict, dataset: str) -> bool:
    applies = task_cfg.get("applies_to_datasets", [])
    return not applies or dataset in applies


def render_item(row: pd.Series, task: str, task_cfg: dict, phrasing: str,
                templates: dict, system_prompt: str) -> list[PromptRow]:
    """The MC and OE prompts for one (slice, task, phrasing)."""
    task_templates = templates.get(task, {})
    if not task_templates:
        return []

    image_id = row["image_id"]
    answer_space = [str(a) for a in task_cfg.get("answer_space", [])]
    aliases = {str(k): [str(v) for v in (vs or [])]
               for k, vs in (task_cfg.get("answer_space_aliases") or {}).items()}
    label = reference_label(row, task)

    rendered: list[PromptRow] = []
    for fmt in ("mc", "oe"):
        template = task_templates.get(fmt, "")
        if not template or (fmt == "mc" and not answer_space):
            continue

        if fmt == "mc":
            options, order = shuffle_options(answer_space, item_seed(image_id, task, phrasing))
            text = template.format(options=format_options(options))
            letter = correct_letter(options, label, aliases)
        else:
            options, order, letter = [], [], ""
            text = template.format(options="")

        rendered.append(PromptRow(
            prompt_id=f"{image_id}__{task}__{phrasing}__{fmt}",
            image_id=image_id,
            task=task,
            phrasing=phrasing,
            format=fmt.upper(),
            prompt_text=text.strip(),
            system_prompt=system_prompt.strip(),
            mc_options=options,
            mc_option_order=order,
            mc_correct_letter=letter,
        ))
    return rendered


def render_all(prompts_cfg: dict, tasks_cfg: dict, manifest: pd.DataFrame,
               phrasings: list[str] | None = None) -> list[PromptRow]:
    phrasings = phrasings or list(prompts_cfg.get("phrasings", {}))
    system_prompt = prompts_cfg.get("system_prompt", "")

    rows: list[PromptRow] = []
    for _, item in manifest.iterrows():
        dataset = item.get("dataset", "")
        for task, task_cfg in tasks_cfg.get("tasks", {}).items():
            if not task_applies(task_cfg, dataset):
                continue
            for phrasing in phrasings:
                rows.extend(render_item(
                    item, task, task_cfg, phrasing,
                    prompts_cfg.get("phrasings", {}).get(phrasing, {}), system_prompt,
                ))
    log.info(f"rendered {len(rows)} prompts over {len(manifest)} slices")
    return rows

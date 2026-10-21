"""Rendering the prompt matrix, and the option shuffle that grading depends on."""
from __future__ import annotations

import pandas as pd
import pytest

from src.prompts import render
from src.utils import load_yaml, shuffle_options

SLICE = {"image_id": "brats__X__T2__axial__0p40", "dataset": "brats",
         "label_modality": "T2", "label_plane": "axial",
         "label_tumor_present": "yes", "label_laterality": "left"}


@pytest.fixture
def configs():
    return load_yaml("config/prompts.yaml"), load_yaml("config/tasks.yaml")


def test_the_same_item_always_gets_the_same_option_order():
    seed = render.item_seed("img", "T4-TUMOR", "neutral")
    assert shuffle_options(["no", "yes"], seed) == shuffle_options(["no", "yes"], seed)


def test_different_items_do_not_share_a_seed():
    assert (render.item_seed("a", "T4-TUMOR", "neutral")
            != render.item_seed("b", "T4-TUMOR", "neutral"))
    assert (render.item_seed("a", "T4-TUMOR", "neutral")
            != render.item_seed("a", "T4-TUMOR", "terse"))


def test_the_correct_letter_follows_the_shuffle():
    assert render.correct_letter(["no", "yes"], "yes", {}) == "B"
    assert render.correct_letter(["yes", "no"], "yes", {}) == "A"


def test_an_alias_still_finds_its_option():
    aliases = {"T1ce": ["t1c", "post-contrast"]}
    assert render.correct_letter(["T1", "T1ce"], "t1c", aliases) == "B"


def test_a_label_outside_the_answer_space_is_flagged():
    assert render.correct_letter(["no", "yes"], "maybe", {}) == "?"


def test_every_rendered_multiple_choice_prompt_has_a_known_answer(configs):
    prompts_cfg, tasks_cfg = configs
    rows = render.render_all(prompts_cfg, tasks_cfg, pd.DataFrame([SLICE]), ["neutral"])
    multiple_choice = [r for r in rows if r["format"] == "MC"]
    assert multiple_choice
    assert all(r["mc_correct_letter"] != "?" for r in multiple_choice)


def test_the_tumor_prompt_asks_the_question_the_paper_quotes(configs):
    prompts_cfg, tasks_cfg = configs
    rows = render.render_all(prompts_cfg, tasks_cfg, pd.DataFrame([SLICE]), ["neutral"])
    tumor = next(r for r in rows if r["task"] == "T4-TUMOR" and r["format"] == "MC")
    assert tumor["prompt_text"].startswith(
        "Does this brain MRI image show a tumor or gross abnormality?")
    assert "Confidence: <0-100>" in tumor["prompt_text"]


def test_a_task_is_not_asked_of_a_dataset_it_does_not_apply_to(configs):
    prompts_cfg, tasks_cfg = configs
    healthy = {**SLICE, "dataset": "ixi", "image_id": "ixi__IXI001__T1__axial__0p40"}
    rows = render.render_all(prompts_cfg, tasks_cfg, pd.DataFrame([healthy]), ["neutral"])
    # Laterality is a BraTS-only question.
    assert not any(r["task"] == "T5-LAT" for r in rows)

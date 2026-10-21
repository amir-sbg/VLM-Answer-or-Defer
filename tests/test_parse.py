"""Parsing an answer and a stated confidence out of a raw response."""
from __future__ import annotations

import pytest

from src.scoring import parse

YES_NO = ["no", "yes"]


def parse_mc(text: str, options=None, reasoning: bool = False) -> dict:
    options = options or YES_NO
    return parse.parse_response(text, "MC", options, options, reasoning=reasoning)


# ── the ordinary case ─────────────────────────────────────────────────────────

def test_a_letter_and_a_confidence():
    result = parse_mc("B\nConfidence: 95")
    assert result["pred_answer"] == "yes" and result["confidence"] == 95
    assert not result["unparseable"]


@pytest.mark.parametrize("text,expected", [
    ("A", "no"),
    ("A.", "no"),
    ("B) yes", "yes"),
    ("Answer: B", "yes"),
    ("The answer is A", "no"),
    ("yes", "yes"),
])
def test_the_forms_a_choice_can_take(text, expected):
    assert parse_mc(f"{text}\nConfidence: 80")["pred_answer"] == expected


@pytest.mark.parametrize("text,expected", [
    ("Confidence: 0", 0),
    ("Confidence: 100", 100),
    ("confidence 73", 73),
    ("Confidence: 85/100", 85),
    ("Confidence: 85%", 85),
    ("My confidence is about 60", 60),
])
def test_the_forms_a_confidence_can_take(text, expected):
    assert parse.extract_confidence(text) == expected


def test_a_confidence_outside_the_scale_is_not_one():
    assert parse.extract_confidence("Confidence: 150") is None


# ── the cases that would otherwise invent an answer ───────────────────────────

def test_a_word_beginning_with_an_option_letter_is_not_that_option():
    """"Bilateral" must not be read as option B."""
    options = ["left", "right", "bilateral", "none"]
    assert parse_mc("Bilateral involvement.\nConfidence: 70", options)["pred_answer"] == "bilateral"


def test_a_reasoning_model_is_read_only_after_its_trace():
    text = "<think>Option A looks right, no wait.</think>\nB\nConfidence: 90"
    assert parse_mc(text, reasoning=True)["pred_answer"] == "yes"


def test_a_truncated_trace_yields_no_answer_at_all():
    result = parse_mc("<think>weighing the evidence and running out of room",
                      reasoning=True)
    assert result["unparseable"] and result["truncated_trace"]
    assert result["pred_answer"] == "UNPARSEABLE"


def test_a_non_reasoning_model_is_read_whole():
    assert parse_mc("B\nConfidence: 90", reasoning=False)["pred_answer"] == "yes"


def test_decoded_special_tokens_are_stripped():
    text = "<|begin_of_box|>A\nConfidence: 80<|end_of_box|>"
    result = parse_mc(text)
    assert result["pred_answer"] == "no" and result["confidence"] == 80


# ── abstention and failure ────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "UNSURE",
    "I cannot determine whether a tumor is present.",
    "This is not a brain MRI.",
    "The image quality is too poor to interpret.",
])
def test_an_abstention_is_not_an_answer(text):
    result = parse_mc(text)
    assert result["abstained"] and result["pred_answer"] == "UNSURE"


def test_an_unreadable_response_is_marked_as_such():
    assert parse_mc("hmm")["unparseable"]


def test_an_empty_response_is_unparseable():
    assert parse_mc("")["unparseable"]


# ── open-ended ────────────────────────────────────────────────────────────────

def test_open_ended_text_is_matched_against_the_answer_space():
    result = parse.parse_response(
        "There is no evidence of a tumor.\nConfidence: 85", "OE", YES_NO)
    assert result["pred_answer"] == "no" and result["confidence"] == 85


def test_longer_aliases_win_over_shorter_ones():
    assert parse.normalize_answer("post-contrast T1", ["T1", "T1ce", "T2"]) == "T1ce"


def test_reasoning_models_are_recognised_by_id():
    assert parse.is_reasoning_model("Qwen/Qwen3.6-27B")
    assert parse.is_reasoning_model("zai-org/GLM-4.6V-Flash")
    assert not parse.is_reasoning_model("google/gemma-4-12B-it")
    assert not parse.is_reasoning_model("")

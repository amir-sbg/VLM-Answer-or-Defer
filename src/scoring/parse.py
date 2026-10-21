"""Turn a raw model response into an answer and a stated confidence.

Three decisions here shape every number in the paper.

1. For a reasoning model, only the text after the final ``</think>`` counts.
   A trace that hit the token limit before answering has no such text, and
   mining the trace for a stray letter would invent an answer the model never
   gave. Those responses are unparseable, and an unparseable response is
   counted as deferred -- it costs coverage, not accuracy.

2. Confidence is the first integer in 0..100 after the word "Confidence",
   rescaled to [0, 1]. A response with an answer but no parseable confidence
   cannot be put on either side of a threshold, so it is deferred too.

3. A letter is only read as a choice where it is unambiguously one: at the
   start of the answer, or after "Answer:". Otherwise "Bilateral" would be
   read as option B.
"""
from __future__ import annotations

import re
from typing import Any

# Spellings a model may use instead of the exact option label.
ANSWER_ALIASES: dict[str, list[str]] = {
    "T1": ["t1", "t1w", "t1-weighted", "t1weighted", "t1 weighted"],
    "T1ce": ["t1ce", "t1c", "t1-ce", "t1 post-contrast", "post-contrast", "post contrast",
             "contrast-enhanced t1", "t1 with contrast", "t1 gad", "gadolinium-enhanced"],
    "T2": ["t2", "t2w", "t2-weighted", "t2weighted", "t2 weighted"],
    "FLAIR": ["flair", "fluid attenuated", "fluid-attenuated inversion recovery"],
    "PD": ["pd", "proton density", "proton-density", "pd weighted"],
    "MRA": ["mra", "magnetic resonance angiography", "angiography", "angio"],
    "DWI": ["dwi", "diffusion", "diffusion weighted", "diffusion-weighted"],
    "axial": ["axial", "transverse", "transaxial", "horizontal", "cross-sectional"],
    "sagittal": ["sagittal", "lateral", "midsagittal", "mid-sagittal"],
    "coronal": ["coronal", "frontal", "anterior"],
    "yes": ["yes", "present", "detected", "visible", "there is", "shows", "demonstrates",
            "abnormal", "confirms", "consistent with", "positive"],
    "no": ["no", "absent", "not detected", "normal", "not present", "none", "negative",
           "unremarkable", "without", "no evidence"],
    "left": ["left", "left hemisphere", "left-sided", "left side", "lh"],
    "right": ["right", "right hemisphere", "right-sided", "right side", "rh"],
    "bilateral": ["bilateral", "both hemispheres", "both sides", "bihemispheric", "diffuse"],
    "none": ["none", "n/a", "not applicable", "no abnormality", "no lesion"],
}

_ALIAS_TO_LABEL: dict[str, str] = {}
for _label, _aliases in ANSWER_ALIASES.items():
    _ALIAS_TO_LABEL[_label.lower()] = _label
    for _alias in _aliases:
        _ALIAS_TO_LABEL[_alias.lower()] = _label

_CONFIDENCE_PATTERNS = [
    re.compile(r"confidence[:\s]+(\d{1,3})\s*(?:/\s*100|%)?", re.IGNORECASE),
    re.compile(r"(\d{1,3})\s*%?\s*confidence", re.IGNORECASE),
    re.compile(r"my confidence.*?(\d{1,3})", re.IGNORECASE),
    re.compile(r"certainty[:\s]+(\d{1,3})", re.IGNORECASE),
]

_ABSTAIN_PATTERNS = [
    re.compile(r"\bUNSURE\b", re.IGNORECASE),
    re.compile(r"\bI cannot\b.*\b(determine|identify|assess)\b", re.IGNORECASE),
    re.compile(r"\b(not\s+a\s+brain|not\s+suitable|non-diagnostic|uninterpretable"
               r"|cannot\s+be\s+evaluated|too\s+poor|insufficient\s+quality)\b", re.IGNORECASE),
    re.compile(r"\b(unable to|not able to)\b.*\b(assess|determine|identify|interpret)\b",
               re.IGNORECASE),
    re.compile(r"\bI\s+don[\'']t\s+have\s+enough\b", re.IGNORECASE),
    re.compile(r"\bI\s+cannot\s+provide\b", re.IGNORECASE),
]

# Some models wrap the answer in decoded special-token markers, for instance
# GLM's "<|begin_of_box|>B\nConfidence: 95<|end_of_box|>".
_SPECIAL_MARKER = re.compile(r"<\|.*?\|>")

REASONING_MODEL_IDS = {
    "zai-org/GLM-4.6V-Flash",
    "Qwen/Qwen3.6-27B",
}


def is_reasoning_model(model_id: str) -> bool:
    if not model_id:
        return False
    return model_id in REASONING_MODEL_IDS or "reasoning" in model_id.lower()


def answer_segment(text: str, reasoning: bool) -> str | None:
    """The part of the response that holds the answer, or None if there is none."""
    if "</think>" in text:
        segment = text.rsplit("</think>", 1)[-1]
    elif reasoning:
        return None           # trace ran out of tokens before it answered
    else:
        segment = text
    return _SPECIAL_MARKER.sub(" ", segment).strip()


def extract_confidence(text: str) -> int | None:
    for pattern in _CONFIDENCE_PATTERNS:
        match = pattern.search(text)
        if match:
            value = int(match.group(1))
            if 0 <= value <= 100:
                return value
    return None


def detect_abstention(text: str) -> bool:
    return any(pattern.search(text) for pattern in _ABSTAIN_PATTERNS)


def normalize_answer(text: str, answer_space: list[str]) -> str | None:
    """Map free text onto the task's answer space, or None."""
    if not text or not answer_space:
        return None

    cleaned = re.sub(r"^[A-Z]\.\s*", "", text.strip()).strip().strip("*").strip().lower()
    by_lower = {a.lower(): a for a in answer_space}

    if cleaned in by_lower:
        return by_lower[cleaned]

    label = _ALIAS_TO_LABEL.get(cleaned)
    if label and label in answer_space:
        return label

    # Longest alias first, so "no evidence" is not matched by "no".
    for alias, label in sorted(_ALIAS_TO_LABEL.items(), key=lambda kv: -len(kv[0])):
        if alias in cleaned and label in answer_space:
            return label

    for option in sorted(answer_space, key=len, reverse=True):
        if option.lower() in cleaned:
            return option
    return None


def extract_mc_answer(text: str, options: list[str]) -> str | None:
    """The option a multiple-choice response selected."""
    if not text or not options:
        return None
    cleaned = text.strip()

    match = re.search(r"(?:answer\s*(?:is|:)?\s*)([A-Z])\b", cleaned, re.IGNORECASE)
    if not match:
        # A bare letter at the very start, followed by a separator or nothing:
        # "B", "B.", "B) yes" -- but not "Bilateral".
        match = re.match(r"^([A-Z])(?:[.\):,]|\s|$)", cleaned)
    if match:
        index = ord(match.group(1).upper()) - ord("A")
        if 0 <= index < len(options):
            return options[index]

    return normalize_answer(cleaned, options)


def parse_response(response_text: str, format_: str, answer_space: list[str],
                   mc_options: list[str] | None = None,
                   reasoning: bool = False) -> dict[str, Any]:
    """Parse one response into answer, confidence, abstention and parse status."""
    segment = answer_segment(response_text or "", reasoning)
    if segment is None:
        return {"pred_answer": "UNPARSEABLE", "confidence": None,
                "abstained": False, "unparseable": True, "truncated_trace": True}

    confidence = extract_confidence(segment)
    if detect_abstention(segment):
        return {"pred_answer": "UNSURE", "confidence": confidence,
                "abstained": True, "unparseable": False, "truncated_trace": False}

    if format_ == "MC" and mc_options:
        answer = extract_mc_answer(segment, mc_options)
    else:
        answer = normalize_answer(segment, answer_space)

    return {"pred_answer": answer or "UNPARSEABLE", "confidence": confidence,
            "abstained": False, "unparseable": answer is None, "truncated_trace": False}


def parse_record(record: dict, tasks_cfg: dict) -> dict:
    """Parse one transcript record, returning it with the parsed fields added."""
    task_cfg = tasks_cfg.get("tasks", {}).get(record.get("task", ""), {})
    parsed = parse_response(
        response_text=record.get("response", ""),
        format_=record.get("format", "OE"),
        answer_space=[str(a) for a in task_cfg.get("answer_space", [])],
        mc_options=[str(o) for o in record.get("mc_options", [])],
        reasoning=is_reasoning_model(record.get("model", "")),
    )
    return {**record, **parsed}


def parse_batch(records: list[dict], tasks_cfg: dict) -> list[dict]:
    return [parse_record(r, tasks_cfg) for r in records]

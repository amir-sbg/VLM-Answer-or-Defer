"""Figure 2: one exchange end to end, with four real slices.

Nothing in this figure is drawn by hand. The images are the exact 512x512
letterboxed PNGs the model was shown, uncropped. The red outlines are
re-derived from the BraTS segmentation through the same reorientation, slice
index and letterbox geometry used to write those PNGs, resampled
nearest-neighbour, so the outline marks the pixels the reference actually
covers. The prompt is rendered from the config with the options in the order
the first panel saw them, and the response is verbatim from the transcript.

The four panels are chosen by a stated rule and a fixed seed, not by eye. Each
cell names a condition; among the slices that satisfy it, one is drawn at
random with seed 0. Lesion areas printed in the titles are the native mask
areas from the manifest (1 px = 1 mm^2 on the BraTS grid), not the pixel counts
of the upscaled image.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src.data.slice_extract import letterbox_resize
from src.data.slice_labels import mask_slice
from src.figures.style import INK, TEXT_WIDTH, save, use_paper_style
from src.utils import get_logger, project_path

log = get_logger(__name__)

RIGHT, WRONG = "#1a9850", "#d73027"
MASK = "#e41a1c"
BOX_FILL = {"prompt": "#ece2fb", "raw": "#fdf3cd", "parsed": "#dcecfb"}
# A very small mask is a few pixels at print size; ringing it keeps it findable.
RING_BELOW_PX = 300


@dataclass(frozen=True)
class PanelRule:
    """One cell of the figure: a caption and the condition slices must satisfy."""
    key: str
    caption: str
    stratum: str
    answer: int
    min_confidence: float = 0.0
    min_area_px: float | None = None
    max_area_px: float | None = None


DEFAULT_RULES = (
    PanelRule("ixi_fp", "(a) IXI", "IXI", answer=1, min_confidence=0.90),
    PanelRule("brats_fp", "(b) BraTS−", "BraTS-", answer=1, min_confidence=0.90),
    PanelRule("missed", "(c) BraTS+", "BraTS+", answer=0,
              min_confidence=0.85, max_area_px=100),
    PanelRule("found", "(d) BraTS+", "BraTS+", answer=1,
              min_confidence=0.0, min_area_px=1600),
)


def choose_panels(table: pd.DataFrame, model: str, rules=DEFAULT_RULES,
                  plane: str = "axial", seed: int = 0) -> pd.DataFrame:
    """One slice per rule, drawn at random from the slices that qualify."""
    rows = table[(table["model_label"] == model) & (table["plane"] == plane)
                 & table["answered"]]
    rng = np.random.default_rng(seed)

    chosen = []
    for rule in rules:
        eligible = rows[(rows["stratum"] == rule.stratum)
                        & (rows["answer"] == rule.answer)
                        & (rows["confidence"] >= rule.min_confidence)]
        if rule.min_area_px is not None:
            eligible = eligible[eligible["lesion_area_px"] >= rule.min_area_px]
        if rule.max_area_px is not None:
            eligible = eligible[eligible["lesion_area_px"] < rule.max_area_px]
        if eligible.empty:
            log.warning(f"no slice satisfies panel rule {rule.key}")
            continue
        pick = eligible.sort_values("image_id").iloc[rng.integers(len(eligible))]
        chosen.append({**pick.to_dict(), "panel": rule.key, "caption": rule.caption,
                       "n_eligible": len(eligible)})
    return pd.DataFrame(chosen)


def mask_for(image_id: str, provenance: dict[str, dict]) -> np.ndarray | None:
    """The reference outline for a slice, aligned with the displayed PNG."""
    record = provenance.get(image_id)
    if record is None or record.get("dataset") != "brats":
        return None
    raw = mask_slice(str(project_path(record["volume_path"])),
                     record["plane"], float(record["slice_frac"]))
    if raw is None:
        return None
    return letterbox_resize(raw, tuple(record.get("output_size", (512, 512))), nearest=True)


def draw_slice(ax, image: np.ndarray, mask: np.ndarray | None, title: str) -> None:
    ax.imshow(image, cmap="gray", interpolation="nearest", aspect="equal")
    if mask is not None and mask.any():
        ax.contour(mask > 0, levels=[0.5], colors=MASK, linewidths=1.0)
        if int(mask.sum()) < RING_BELOW_PX:
            ys, xs = np.nonzero(mask)
            radius = max(26.0, 2.5 * np.sqrt(mask.sum() / np.pi))
            ax.add_patch(plt.Circle((xs.mean(), ys.mean()), radius, fill=False,
                                    ec=MASK, lw=0.8, ls=(0, (3, 2))))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#999999")
        spine.set_linewidth(0.5)
    ax.set_title(title, fontsize=6.6, loc="left", pad=3)


def draw_text_boxes(ax, blocks: list[tuple[str, str, str]], body_size: float = 4.6,
                    title_size: float = 5.8, wrap: int = 37) -> None:
    """Stack the prompt, the raw response and the parsed result down one column.

    Heights are measured in points and converted to axes fractions, rather than
    guessed as fractions directly: a guess that fits at one figure size silently
    overlaps at another, and these boxes sit next to images whose aspect fixes
    the axis height.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    axis_points = ax.get_window_extent().height / ax.figure.dpi * 72.0
    line_pt = body_size * 1.32
    title_pt = title_size * 1.9
    pad_pt = 3.0
    gap_pt = 4.0

    wrapped = [(title, "\n".join(textwrap.fill(line, wrap) if line else ""
                                 for line in body.split("\n")), kind)
               for title, body, kind in blocks]
    heights_pt = [title_pt + line_pt * (body.count("\n") + 1) + pad_pt
                  for _, body, _ in wrapped]

    total_pt = sum(heights_pt) + gap_pt * (len(heights_pt) - 1)
    if total_pt > axis_points:
        # Too tall for the column: shrink everything by one factor so the type
        # keeps its relative sizes.
        scale = axis_points / total_pt
        body_size, title_size = body_size * scale, title_size * scale
        heights_pt = [h * scale for h in heights_pt]
        title_pt, gap_pt = title_pt * scale, gap_pt * scale

    to_axes = lambda pt: pt / axis_points
    top = 1.0
    for (title, body, kind), height_pt in zip(wrapped, heights_pt):
        height = to_axes(height_pt)
        ax.add_patch(FancyBboxPatch((0.01, top - height), 0.98, height,
                                    boxstyle="round,pad=0.003,rounding_size=0.010",
                                    fc=BOX_FILL[kind], ec="#6f6f6f", lw=0.5,
                                    clip_on=False))
        ax.text(0.035, top - to_axes(2.0), title, fontsize=title_size,
                fontweight="bold", va="top")
        ax.text(0.035, top - to_axes(title_pt), body, fontsize=body_size, va="top",
                family="monospace", linespacing=1.25)
        top -= height + to_axes(gap_pt)


def abridged_prompt(prompt_text: str) -> str:
    return ("System: [...] After answering, always state your confidence as an integer "
            "from 0 to 100: Confidence: <number>. If the image is unsuitable [...], "
            "respond with UNSURE.\n\nUser: " + prompt_text)


def make(panels: pd.DataFrame, provenance: dict[str, dict], prompt_text: str,
         raw_response: str, parsed_caption: str,
         path: str = "figures/fig2_example.pdf"):
    use_paper_style()
    n = len(panels)
    fig, axes = plt.subplots(1, n + 1, figsize=(TEXT_WIDTH, 2.35),
                             gridspec_kw={"width_ratios": [1.5] + [1.0] * n,
                                          "wspace": 0.06})
    # The text column is laid out in points, so it has to know its own size.
    fig.canvas.draw()

    draw_text_boxes(axes[0], [
        ("Prompt (abridged)", abridged_prompt(prompt_text), "prompt"),
        ("Raw response", raw_response, "raw"),
        ("Parsed", parsed_caption, "parsed"),
    ])

    for ax, (_, row) in zip(axes[1:], panels.iterrows()):
        image = plt.imread(str(project_path(row["path"])))
        if image.ndim == 3:
            image = image[..., :3].mean(axis=2)
        title = row["caption"]
        if row["stratum"] == "BraTS+":
            title += f", {int(row['lesion_area_px']):,} px"
        if len(title) > 22:
            title = title.replace(", ", ",\n", 1)
        draw_slice(ax, image, mask_for(row["image_id"], provenance), title)

        correct = bool(row["correct"])
        answer = "yes" if row["answer"] == 1 else "no"
        mark = r"$\checkmark$" if correct else r"$\times$"
        ax.set_xlabel(f"reference: {'yes' if row['reference'] else 'no'}\n"
                      f"answer: {answer}, $c$ = {row['confidence']:.2f}   {mark}",
                      fontsize=6.2, color=RIGHT if correct else WRONG, labelpad=4)

    return save(fig, path)

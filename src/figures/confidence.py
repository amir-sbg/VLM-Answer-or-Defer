"""Figure 3: confidence differs more between answers than between right and wrong.

Panel (a) is drawn as a dot plot rather than bars. Every value lies between
0.78 and 0.97 and the claim is that four of them are nearly *equal*, so the
axis has to start well above zero to show anything -- and a bar whose baseline
is not zero misstates the ratio between its neighbours. A dot encodes position
only, so a cropped axis is honest, and the thing the panel is about, the
vanishing gap between a true and a false positive, is what you see first.

Panel (b) pairs each model's correlation on the two populations and draws the
connector between them, so the eye reads the move rather than two separate
columns of points.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures.style import (COLUMN_WIDTH, INK, MUTED, POPULATION_COLORS,
                               POPULATION_MARKERS, TEXT_WIDTH, color_for, save, tidy,
                               use_paper_style)

OUTCOMES = ["true_positive", "false_positive", "true_negative", "false_negative"]
OUTCOME_LABELS = ["true pos.", "false pos.", "true neg.", "false neg."]


def confidence_by_outcome(ax, outcome_confidence: pd.DataFrame, models: list[str],
                          order: list[str]) -> None:
    wide = (outcome_confidence.pivot(index="model", columns="outcome",
                                     values="mean_confidence")
            .reindex(index=models, columns=OUTCOMES))
    x = np.arange(len(OUTCOMES))
    offsets = np.linspace(-0.22, 0.22, len(models))

    # Faint separators between the outcome groups, so the eye does not read a
    # dot as belonging to the neighbouring category.
    for boundary in x[:-1] + 0.5:
        ax.axvline(boundary, color="#e8e8e8", linewidth=0.5, zorder=1)

    for model, offset in zip(models, offsets):
        values = wide.loc[model].to_numpy(float)
        ax.plot(x + offset, values, linestyle="none", marker="o", markersize=4.4,
                color=color_for(model, order), markeredgecolor="white",
                markeredgewidth=0.6, label=model, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(OUTCOME_LABELS)
    ax.set_ylabel("mean stated confidence")
    ax.set_xlim(-0.5, len(OUTCOMES) - 0.5)
    tidy(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              handletextpad=0.3, columnspacing=1.2)


def correlation_by_population(ax, rho: pd.DataFrame, models: list[str]) -> None:
    wide = rho.pivot(index="model", columns="population", values="rho").reindex(models)
    # Largest move at the top, so the figure's own claim orders the rows.
    wide = wide.assign(gap=(wide["IXI"] - wide["BraTS"]).abs()).sort_values("gap")
    y = np.arange(len(wide))

    ax.axvline(0.0, color=MUTED, linewidth=0.6, zorder=1)
    for i, (_, row) in enumerate(wide.iterrows()):
        left, right = sorted((("BraTS", row["BraTS"]), ("IXI", row["IXI"])),
                             key=lambda pair: pair[1])
        ax.plot([left[1], right[1]], [i, i], color="#b8b8b8", linewidth=1.0, zorder=2)
        for population in ("BraTS", "IXI"):
            ax.plot(row[population], i, linestyle="none",
                    marker=POPULATION_MARKERS[population], markersize=4.4,
                    color=POPULATION_COLORS[population], markeredgecolor="white",
                    markeredgewidth=0.6, zorder=3,
                    label=population if i == 0 else None)

        # Each value sits on the outside of its own marker. Where the two marks
        # nearly coincide there is no outside, so the labels stack instead.
        crowded = abs(right[1] - left[1]) < 0.22
        for (population, value), side in ((left, -1), (right, +1)):
            ax.annotate(
                f"{value:+.2f}",
                (value + side * 0.025, i + (0.30 if crowded and side > 0 else
                                            -0.30 if crowded and side < 0 else 0.0)),
                ha="left" if side > 0 else "right", va="center", fontsize=5.6,
                color=POPULATION_COLORS[population])

    ax.set_yticks(y)
    ax.set_yticklabels(wide.index)
    ax.set_xlabel(r"correlation $\rho$ between stated confidence and correctness")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.7, len(wide) - 0.3)
    tidy(ax, grid_axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              handletextpad=0.3, columnspacing=1.6)


def make(outcome_confidence: pd.DataFrame, rho: pd.DataFrame, order: list[str],
         outcome_models: list[str] | None = None, path: str = "figures/fig3_confidence.pdf"):
    """Both panels side by side, at full text width."""
    use_paper_style()
    outcome_models = outcome_models or order

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.7),
                             gridspec_kw={"width_ratios": [1.0, 1.3], "wspace": 0.30})
    axes[0].set_ylim(0.74, 1.005)
    confidence_by_outcome(axes[0], outcome_confidence, outcome_models, order)
    correlation_by_population(axes[1], rho, order)

    for ax, title in zip(axes, ("(a) confidence by outcome",
                                "(b) same model, opposite sign by population")):
        ax.set_title(title, loc="left", color=INK, pad=4)
    return save(fig, path)

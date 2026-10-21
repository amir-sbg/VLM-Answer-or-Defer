"""Figure 4: the error rate of a model's favoured answers against the case mix.

Eight series on one axis, so colour is not asked to carry the identity alone:
the line style splits them by which answer the model states more confidently,
which is also the thing that decides whether a curve rises or falls with pi.
Horizontal rules mark the error budgets and a vertical rule the natural BraTS
mix, so a reader can see at a glance which curves pass which budget where.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.figures.style import (COLUMN_WIDTH, MUTED, color_for, save, tidy, use_paper_style)

BUDGET_STYLE = {"color": MUTED, "linewidth": 0.5, "linestyle": (0, (1, 2))}


def make(curves: pd.DataFrame, natural_pi: float, alphas: list[float],
         order: list[str], path: str = "figures/fig4_case_mix.pdf"):
    use_paper_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.6))

    # The budgets are close together near zero, so the lowest one is labelled
    # below its rule and the rest above, which is what keeps them apart.
    for i, alpha in enumerate(sorted(alphas)):
        ax.axhline(alpha, **BUDGET_STYLE, zorder=1)
        below = i == 0
        ax.annotate(f"{alpha:.0%}", (0.995, alpha), xytext=(0, -2 if below else 2),
                    textcoords="offset points", ha="right",
                    va="top" if below else "bottom", fontsize=5.6, color=MUTED)
    ax.axvline(natural_pi, color="#9a9a9a", linewidth=0.6, zorder=1)
    ax.annotate("BraTS", (natural_pi, 1.0), xytext=(2, -1), textcoords="offset points",
                ha="left", va="top", fontsize=5.8, color=MUTED)

    for model in order:
        part = curves[curves["model"] == model]
        if part.empty:
            continue
        favoured = part["more_confident_answer"].iloc[0]
        ax.plot(part["pi"], part["error_rate"], color=color_for(model, order),
                linewidth=1.2, linestyle="-" if favoured == "yes" else (0, (4, 1.6)),
                label=f"{model} ({favoured})", zorder=3)

    ax.set_xlabel(r"share of reference-positive slices $\pi$")
    ax.set_ylabel("error rate of more-confident answers")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    tidy(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=2,
              handlelength=1.8, handletextpad=0.4, columnspacing=0.9)
    return save(fig, path)

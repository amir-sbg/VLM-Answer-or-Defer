"""Figure 5: sensitivity against the in-plane area of the annotated lesion.

The x axis is ordinal, four area bins, so the points sit at even spacing and
the connecting line is there to follow a model across bins rather than to
assert a rate of change. Each model gets its own marker as well as its own
colour, which is what keeps the eight series separable in print and under
colour-vision deficiency.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures.style import (COLUMN_WIDTH, color_for, marker_for, save, tidy,
                               use_paper_style)


def make(by_area: pd.DataFrame, order: list[str], path: str = "figures/fig5_lesion_area.pdf"):
    use_paper_style()
    bins = list(dict.fromkeys(by_area["area_bin"]))
    x = np.arange(len(bins))

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3))
    for model in order:
        part = by_area[by_area["model"] == model].set_index("area_bin").reindex(bins)
        ax.plot(x, part["sensitivity"].to_numpy(float), color=color_for(model, order),
                marker=marker_for(model, order), markersize=3.6, linewidth=1.1,
                markeredgecolor="white", markeredgewidth=0.4, label=model, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel("in-plane lesion area on the slice (px)")
    ax.set_ylabel("sensitivity (BraTS+)")
    ax.set_xlim(-0.2, len(bins) - 0.8)
    ax.set_ylim(0, 1)
    tidy(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              handlelength=1.8, handletextpad=0.4, columnspacing=0.9)
    return save(fig, path)

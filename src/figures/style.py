"""Shared figure styling: page geometry, type, palette, and save helpers.

Every figure is vector PDF at the exact width it occupies in the paper, with
fonts embedded and type matching the body text, so nothing is rescaled in
LaTeX and no label ends up a different size from its neighbours.

On colour. Eight models is at the top of what one set of categorical hues can
carry. The palette below sits inside the OKLCH lightness band 0.45-0.74 with
chroma at or above 0.11, and every one of them clears a 3:1 contrast ratio
against white. Neighbouring pairs are at least 17 apart in OKLab dE for normal
vision and at least 15 apart under simulated protanopia and deuteranopia.

No eight-hue set separates *every* pair under colour-vision deficiency -- the
red-green axis collapses, and the worst pair here is 4 apart -- so hue is never
the only thing distinguishing two series: the case-mix figure also splits them
by line style, the lesion-area figure by marker, and the outcome panel by
position within each group. Colour is assigned to a model by its position in
the registry and is never cycled or reassigned when a figure plots a subset,
so a model keeps its colour across figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Column and text widths of the WACV two-column layout, in inches.
COLUMN_WIDTH = 3.25
TEXT_WIDTH = 6.875

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"
SURFACE = "#ffffff"

# Fixed categorical order; index i belongs to the i-th model in config/models.yaml.
MODEL_COLORS = [
    "#c0626c", "#a20a01", "#739100", "#09672e",
    "#0090bb", "#494b96", "#8e5ff2", "#a20d8c",
]
MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Two states of the same quantity, not two categories of model.
POPULATION_COLORS = {"BraTS": "#17558f", "IXI": "#c2560f"}
POPULATION_MARKERS = {"BraTS": "o", "IXI": "s"}


def use_paper_style() -> None:
    """Match the paper's body type and keep axes recessive."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "legend.fontsize": 6.2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def color_for(model: str, order: list[str]) -> str:
    """The colour of a model, by its position in the registry order."""
    return MODEL_COLORS[order.index(model) % len(MODEL_COLORS)]


def marker_for(model: str, order: list[str]) -> str:
    return MODEL_MARKERS[order.index(model) % len(MODEL_MARKERS)]


def tidy(ax, *, grid_axis: str = "y") -> None:
    """Drop the top and right spines and put a faint grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def save(fig, path: str | Path) -> Path:
    """Write the figure as PDF plus a PNG preview, and report the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return path

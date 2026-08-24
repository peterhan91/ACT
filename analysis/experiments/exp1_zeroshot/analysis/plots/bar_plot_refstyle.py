#!/usr/bin/env python3
"""
Grouped bar chart of zero-shot concept-annotation performance, restyled after the
CONCH / pathology-foundation-model Nature figure (the "zero-shot performance" panel):

  * adjacent, flat (no-outline) bars, wider than the MUSK variant;
  * a per-method colored DASHED reference line spanning the full width at that
    model's mean over all four datasets (the figure's signature element);
  * diagonal (30 deg, right-anchored) dataset tick labels;
  * a compact, frameless, multi-column legend with square swatches inside the
    upper plotting area;
  * y-ticks at 0 / 0.25 / 0.50 / 0.75 / 1.00;
  * capless thick black error bars; only left/bottom spines.

`ACT` is the leftmost (standout magenta) bar in every group. A small extra gap
separates the chest pair (CT-RATE, PMBB chest) from the abdominal pair (PMBB
abdominal, RSNA), echoing the reference figure's cluster split.

Uncertainty is CONCEPT-LEVEL: error bars = 95% CI across concepts (1.96 x SEM),
exactly as in bar_plot.py (per-volume probabilities are not stored).

Reads   analysis/tables/per_label_auc__<dataset>.csv
Writes  analysis/plots/figures/bar_zeroshot_refstyle.png

Run:  python plots/bar_plot_refstyle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # analysis/
import config

# ── clean sans-serif look (fall back gracefully if Arial/Helvetica absent) ─────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "svg.fonttype": "none",
    "axes.linewidth": 1.2,
})

DATASETS = ["ctrate_test", "pmbb_chest_nc", "pmbb_abd_ce", "rsna2023_test"]

# Within-group slot order: ACT first, then the complete-cohort 3D and 2D baselines.
SLOT_ORDER = ["ours", "merlin", "ctclip", "medsiglip", "biomedclip", "openai_clip"]

# Palette adopted from the CONCH "zero-shot performance" figure. merlin/biomedclip/
# openai_clip use the EXACT sampled CONCH bar fills (teal-green / gold / sky-blue);
# ctclip/fvlm/medsiglip are harmonized extensions in the same soft register; ours is
# a deeper, more-saturated magenta so the proposed method stays the standout. Defined
# locally so only THIS figure is recolored (config.MODEL_COLORS still drives the
# circular maps). Each method keeps its original color *role* from config.
CONCH_COLORS = {
    "ours":         "#9E2F88",   # deep magenta — standout (hero), CONCH-magenta family
    "merlin":       "#46A99A",   # CONCH teal-green (exact sample)
    "ctclip":       "#70CC7C",   # leaf green (harmonized)
    "fvlm":         "#E68F73",   # soft coral (harmonized; CONCH has no warm tone)
    "medsiglip":    "#9F7ED9",   # soft purple (harmonized)
    "biomedclip":   "#DECB78",   # CONCH gold (exact sample)
    "openai_clip":  "#88CBEE",   # CONCH sky-blue (exact sample)
}

BAR_W = 0.86          # bar width in slot units (<1 -> hairline gap between bars)
GROUP_GAP = 2.0       # blank slots between adjacent dataset groups
SPLIT_GAP = 1.6       # EXTRA blank slots between the chest pair and abdomen pair
SPLIT_AFTER = 1       # index of the last chest group (0=ctrate,1=pmbb_chest)

# How to handle f-VLM, which is chest-only (no bar in the abdominal groups):
#   "drop" -> omit it and re-center the remaining bars, so abdominal groups have
#             no gap (cleanest, matches the CONCH reference). DEFAULT.
#   "na"   -> reserve f-VLM's slot (keeps colors aligned across all groups) and
#             print a small grey "n/a" where its bar would be.
MISSING = "drop"


def _series(dataset: str, model: str) -> np.ndarray:
    df = pd.read_csv(config.TABLES_DIR / f"per_label_auc__{dataset}.csv")
    col = f"auc_{model}"
    return df[col].dropna().to_numpy() if col in df.columns else np.array([])


def _mean_ci(vals: np.ndarray):
    """mean and 95% CI half-width (1.96 * SEM) across concepts."""
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(vals.mean()), 1.96 * float(vals.std(ddof=1)) / np.sqrt(vals.size)


def _dataset_means():
    """{model: mean over datasets of that dataset's mean AUROC} for dashed lines."""
    out = {}
    for m in SLOT_ORDER:
        per = [_mean_ci(_series(d, m))[0] for d in DATASETS]
        per = [v for v in per if v == v]                  # drop NaN (f-VLM abdomen)
        out[m] = float(np.mean(per)) if per else np.nan
    return out


def draw(ax) -> None:
    """Draw the grouped bar chart (with dashed per-method reference lines) on `ax`."""
    n = len(SLOT_ORDER)
    # ── group start x-positions (slot units), with the chest/abdomen split ─────
    starts, x = [], 0.0
    for gi in range(len(DATASETS)):
        starts.append(x)
        x += n + GROUP_GAP + (SPLIT_GAP if gi == SPLIT_AFTER else 0.0)
    centers = [s + (n - 1) / 2.0 for s in starts]

    # ── dashed per-method reference lines (behind bars), at cross-dataset mean ──
    means = _dataset_means()
    for m in SLOT_ORDER:
        if means[m] == means[m]:
            ax.axhline(means[m], color=CONCH_COLORS[m], ls=(0, (6, 4)),
                       lw=1.3, alpha=0.95, zorder=1)

    # ── bars ───────────────────────────────────────────────────────────────────
    ekw = dict(ecolor="black", elinewidth=2.0, capsize=0, zorder=4)
    for gi, d in enumerate(DATASETS):
        if MISSING == "drop":
            # omit absent models; re-center the present bars on the group center
            present = [m for m in SLOT_ORDER if _series(d, m).size]
            k0 = (len(present) - 1) / 2.0
            for k, m in enumerate(present):
                mean, ci = _mean_ci(_series(d, m))
                ax.bar(centers[gi] + (k - k0), mean, BAR_W, yerr=ci,
                       color=CONCH_COLORS[m], edgecolor="none", zorder=3, error_kw=ekw)
        else:  # "na": keep every slot; print "n/a" where a model has no score
            for j, m in enumerate(SLOT_ORDER):
                vals = _series(d, m)
                if vals.size == 0:
                    ax.text(starts[gi] + j, 0.012, "n/a", rotation=90, ha="center",
                            va="bottom", fontsize=9, color="0.55", style="italic")
                    continue
                mean, ci = _mean_ci(vals)
                ax.bar(starts[gi] + j, mean, BAR_W, yerr=ci,
                       color=CONCH_COLORS[m], edgecolor="none", zorder=3, error_kw=ekw)

    # ── axes styling (CONCH look) ───────────────────────────────────────────────
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.2f}"))
    ax.set_ylabel("Mean AUROC", fontsize=15)
    ax.tick_params(axis="y", direction="out", length=4.5, width=1.2, labelsize=13)

    ax.set_xticks(centers)
    ax.set_xticklabels([config.DATASETS[d]["title"] for d in DATASETS],
                       fontsize=13.5)   # horizontal: 4 well-spaced groups fit without tilt
    ax.tick_params(axis="x", length=0, pad=6)

    ax.set_xlim(starts[0] - (1 - BAR_W) - 0.6, starts[-1] + (n - 1) + (1 - BAR_W) + 0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ── compact frameless legend inside the upper plotting area ────────────────
    # Ours first, then 3D baselines, then 2D baselines; square swatches, 4 columns.
    order_3d = [m for m in SLOT_ORDER if m != "ours" and config.MODELS[m]["dim"] == "3D"]
    order_2d = [m for m in SLOT_ORDER if m != "ours" and config.MODELS[m]["dim"] == "2D"]
    legend_order = ["ours"] + order_3d + order_2d

    def _llabel(m):
        disp = config.MODELS[m]["display"]
        return "ACT (3D)" if m == "ours" else f'{disp} ({config.MODELS[m]["dim"]})'

    handles = [Patch(facecolor=CONCH_COLORS[m], edgecolor="none", label=_llabel(m))
               for m in legend_order]
    ax.legend(handles=handles, ncol=3, frameon=False, fontsize=22,
              loc="upper center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.0, handleheight=1.0, columnspacing=1.0,
              handletextpad=0.35, labelspacing=0.45, borderaxespad=0.2)

def render(out_path=None, figsize=(13.5, 6.0), dpi=300) -> Path:
    """Render the bar chart to its own PNG (used standalone and by combined_figure.py)."""
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    out = Path(out_path) if out_path else config.FIGURES_DIR / "bar_zeroshot_refstyle.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    out = render()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

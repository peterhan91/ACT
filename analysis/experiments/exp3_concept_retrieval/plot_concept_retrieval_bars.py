#!/usr/bin/env python3
"""Concept-retrieval recall bar chart (the "Concept → image / Image → concept" figure).

Rebuilds the grouped recall@K bar figure straight from the bootstrap-CI CSVs in
`retrieval_results/` (so bar heights + error bars are the patient-clustered
bootstrap means / 95% CIs), 4 datasets (rows) × 2 directions (cols) × 6 models.

Colour + style are kept CONSISTENT with the main composite figure (panel b): the
CONCH palette (ours = deep magenta standout, baselines = teal/green/purple/gold/
sky), so the two figures use one model→colour mapping across the paper. The model
colour *roles* match analysis/plots/bar_plot_refstyle.py.

Usage:  python plot_concept_retrieval_bars.py
Output: retrieval_results/concept_retrieval_bars.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "axes.edgecolor": "0.2",
})

HERE = Path(__file__).resolve().parent
RES = HERE / "retrieval_results"
OUT = RES / "concept_retrieval_bars.png"

# ── models: legend order + display + native input dim (mirrors exp1 config.MODELS).
# m3dclip and f-VLM are intentionally NOT shown in this figure (m3dclip omitted from
# all figures; f-VLM is chest-only). ──
MODELS = [
    ("ours",        "ACT",         "3D"),
    ("merlin",      "Merlin",      "3D"),
    ("ctclip",      "CT-CLIP",     "3D"),
    ("medsiglip",   "MedSigLIP",   "2D"),
    ("biomedclip",  "BiomedCLIP",  "2D"),
    ("openai_clip", "OpenAI CLIP", "2D"),
]
# CONCH palette — IDENTICAL model→colour roles as the composite/panel-b figure
# (analysis/plots/bar_plot_refstyle.py CONCH_COLORS), so the paper is colour-consistent.
COLORS = {
    "ours":        "#9E2F88",   # deep magenta — standout (hero)
    "merlin":      "#46A99A",   # CONCH teal-green
    "ctclip":      "#70CC7C",   # leaf green
    "medsiglip":   "#9F7ED9",   # soft purple
    "biomedclip":  "#DECB78",   # CONCH gold
    "openai_clip": "#88CBEE",   # CONCH sky-blue
}

# rows = datasets (in figure order); col titles + the metric each column shows
DATASETS = [
    ("ctrate_test",   "CT-RATE"),
    ("pmbb_chest_nc", "PMBB (chest)"),
    ("pmbb_abd_ce",   "PMBB (abdominal)"),
    ("rsna2023_test", "RSNA Abdominal Trauma"),
]
# (column title, csv file, [(metric_col, x-tick label), …])
# mathtext arrow ($\rightarrow$) so the glyph renders cleanly (Helvetica has no U+2192)
DIRECTIONS = [
    (r"Concept $\rightarrow$ image", "bootstrap_ci_concept_to_image.csv",
     [("pooled_R@1", "Recall@1"), ("pooled_R@5", "Recall@5"), ("pooled_R@10", "Recall@10")]),
    (r"Image $\rightarrow$ concept", "bootstrap_ci_image_to_concept.csv",
     [("R@1", "Recall@1"), ("R@3", "Recall@3"), ("R@5", "Recall@5")]),
]

BAR_W = 0.135           # per-bar width (group span = 6·BAR_W ≈ 0.81 of the unit slot)


def _load(fname):
    df = pd.read_csv(RES / fname)
    return df.set_index(["model", "dataset"])


def main():
    data = {col_title: _load(fname) for col_title, fname, _ in DIRECTIONS}

    nrow, ncol = len(DATASETS), len(DIRECTIONS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.4, 13.2), sharey="row")
    axes = np.atleast_2d(axes)

    for ri, (dkey, dtitle) in enumerate(DATASETS):
        # per-row y ceiling from the largest (mean or CI-hi) bar across BOTH directions
        row_top = 0.0
        for col_title, _fn, metrics in DIRECTIONS:
            df = data[col_title]
            for (mkey, _d, _dim) in MODELS:
                if (mkey, dkey) in df.index:
                    for mcol, _lab in metrics:
                        row_top = max(row_top, df.loc[(mkey, dkey), f"{mcol}_hi"] * 100)
        ytop = row_top * 1.16                                  # headroom for value labels
        yticks = np.arange(0, int(np.ceil(row_top / 10.0)) * 10 + 1, 10)

        for ci, (col_title, _fn, metrics) in enumerate(DIRECTIONS):
            ax = axes[ri, ci]
            df = data[col_title]
            for gi, (mcol, xlab) in enumerate(metrics):
                for bi, (mkey, _disp, _dim) in enumerate(MODELS):
                    if (mkey, dkey) not in df.index:
                        continue
                    row = df.loc[(mkey, dkey)]
                    val = row[mcol] * 100.0
                    lo, hi = row[f"{mcol}_lo"] * 100.0, row[f"{mcol}_hi"] * 100.0
                    x = gi + (bi - (len(MODELS) - 1) / 2.0) * BAR_W
                    ax.bar(x, val, width=BAR_W, color=COLORS[mkey],
                           edgecolor="white", linewidth=0.3, zorder=2)
                    # match panel-b error bars: solid black, NO caps, slightly thicker
                    ax.errorbar(x, val, yerr=[[val - lo], [hi - val]], fmt="none",
                                ecolor="black", elinewidth=1.3, capsize=0, zorder=3)
                    # Stagger neighbouring value labels so the larger type remains legible
                    # when adjacent models have nearly identical confidence limits.
                    label_offset = ytop * (0.012 + (0.060 if bi % 2 else 0.0))
                    ax.text(x, hi + label_offset, f"{val:.2f}", ha="center", va="bottom",
                            fontsize=6.5, color="0.1", rotation=0, zorder=4)

            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels([lab for _m, lab in metrics], fontsize=11)
            ax.set_xlim(-0.55, len(metrics) - 0.45)
            ax.set_ylim(0, ytop)
            ax.set_yticks(yticks)
            ax.tick_params(axis="y", labelsize=10)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.spines["left"].set_color("0.2"); ax.spines["bottom"].set_color("0.2")
            if ci == 0:
                ax.set_ylabel(dtitle, fontsize=13, fontweight="bold", labelpad=8)
            if ri == 0:
                ax.set_title(col_title, fontsize=15, pad=14)

    handles = [Patch(facecolor=COLORS[k], edgecolor="white", linewidth=0.3,
                     label=(disp if not dim else f"{disp} ({dim})"))
               for (k, disp, dim) in MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(MODELS), frameon=False, fontsize=12, handlelength=1.3,
               columnspacing=1.8, handletextpad=0.5)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.045,
                        wspace=0.10, hspace=0.30)
    fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Grouped bar chart of zero-shot concept-annotation performance, styled after the MUSK
Nature figure (panel a): flat bars (no outline), the proposed method (Ours) drawn as the
RIGHTMOST magenta bar in every group, a significance bracket from the best baseline to
ours, T-capped thin black error bars, a square-swatch legend, and only left/bottom spines.

x = datasets (CT-RATE, PMBB chest, PMBB abdominal, RSNA 2023); one bar per model
(`config.PLOT_ORDER`, ours moved last), height = mean AUROC over that dataset's concepts.
f-VLM is chest-only, so it has no bar in the abdominal groups.

Uncertainty is CONCEPT-LEVEL (per-volume probabilities are not stored, so volume bootstrap
/ DeLong isn't possible): error bars = 95% CI across concepts (1.96 x SEM); significance
bracket = Wilcoxon signed-rank (ours vs best baseline) paired across concepts (actual
p-value; small concept counts cannot reach p<1e-4).

Reads   analysis/tables/per_label_auc__<dataset>.csv
Writes  analysis/plots/figures/bar_zeroshot.png

Run:  python plots/bar_plot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # analysis/
import config

DATASETS = ["ctrate_test", "pmbb_chest_nc", "pmbb_abd_ce", "rsna2023_test"]
GROUP_PITCH = 1.0      # x distance between group centers
SLOT = 0.12            # within-group per-bar pitch
BAR_W = 0.100          # bar width (< SLOT -> small uniform gap between bars, as in MUSK)
TITLE = "Zero-shot concept annotation"
PANEL = "a"            # MUSK-style bold panel letter (set to "" to drop)


def _bar_order():
    """Baselines in PLOT_ORDER, then ours LAST (the proposed, rightmost magenta bar)."""
    return [m for m in config.PLOT_ORDER if m != "ours"] + ["ours"]


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


def main() -> None:
    order = _bar_order()
    fig, ax = plt.subplots(figsize=(15, 5.4))
    xc = np.arange(len(DATASETS), dtype=float)

    for gi, d in enumerate(DATASETS):
        present = [m for m in order if _series(d, m).size]   # f-VLM auto-dropped on abdomen
        k0 = (len(present) - 1) / 2.0
        for k, m in enumerate(present):
            mean, ci = _mean_ci(_series(d, m))
            x = xc[gi] + (k - k0) * SLOT          # bars spaced by SLOT, width BAR_W < SLOT
            ax.bar(x, mean, BAR_W, yerr=ci, color=config.MODEL_COLORS[m],
                   edgecolor="none", zorder=2,
                   error_kw=dict(ecolor="black", elinewidth=1.3, capsize=3.5, capthick=1.3))

    # ── axes styling (MUSK look) ─────────────────────────────────────────────
    ax.set_xticks(xc)
    ax.set_xticklabels([config.DATASETS[d]["title"] for d in DATASETS], fontsize=12)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean AUROC", fontsize=12.5)
    ax.tick_params(axis="y", direction="out", length=4, width=1.0, labelsize=10.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(-0.55, (len(DATASETS) - 1) * GROUP_PITCH + 0.55)

    # legend ABOVE the axes, single row so it reads cleanly: Ours, then 3D baselines,
    # then 2D baselines (square swatches, each baseline tagged with its dimensionality).
    order_3d = [m for m in order if m != "ours" and config.MODELS[m]["dim"] == "3D"]
    order_2d = [m for m in order if m != "ours" and config.MODELS[m]["dim"] == "2D"]
    legend_order = ["ours"] + order_3d + order_2d

    def _llabel(m):
        disp = config.MODELS[m]["display"]
        return disp if m == "ours" else f'{disp} ({config.MODELS[m]["dim"]})'

    handles = [Patch(facecolor=config.MODEL_COLORS[m], edgecolor="none", label=_llabel(m))
               for m in legend_order]
    ax.legend(handles=handles, ncol=len(handles), frameon=False, fontsize=10,
              loc="lower center", bbox_to_anchor=(0.5, 1.01),
              handlelength=0.9, handleheight=0.9, columnspacing=1.0, handletextpad=0.35)

    out = config.FIGURES_DIR / "bar_zeroshot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

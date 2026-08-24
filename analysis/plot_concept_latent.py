#!/usr/bin/env python3
"""
Replot the "concept-latent" figure from LOCAL concept-UMAP artifacts.

By default, renders only the full 376k-concept UMAP coloured by prespecified
RadLex 4.3-anchored anatomical group, together with its legend.  Pass
``--spotlight`` to restore the optional right-hand 3x3 finding-family grid.
The anatomical and finding label vectors remain intentionally separate because
the spotlight counts and kNN purities were computed from the finding labels.

The cluster pipeline renders two separate PNGs per LLM under
  outputs/concept_umap/<llm>/{concept_latent_v2,concept_latent_spotlight}.png
(synced via tools/hf_sync.py group "figures"). That render script lives on the
cluster; this one rebuilds either the anatomy-only or combined layout from the
local data so the figure can be regenerated / restyled on the Mac. Default LLM
is f2llm.

Inputs (all local, per LLM):
  outputs/concept_umap/<llm>/umap2d.npy       (N,2) float32  2-D UMAP coords
  outputs/concept_umap/<llm>/radlex_anatomy_categories.npy (N,) anatomy groups
Optional with --spotlight:
  outputs/concept_umap/<llm>/categories.npy   (N,)  object   18 groups + 'other'
  outputs/concept_umap/<llm>/knn_purity.json  per-category {purity, n}

Usage:
  python plot_concept_latent.py                 # f2llm -> .../f2llm/concept_latent_anatomy.png
  python plot_concept_latent.py --spotlight     # optional combined layout
  python plot_concept_latent.py --llm openai
  python plot_concept_latent.py --llm f2llm --out /tmp/fig.png --dpi 220
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from radlex_anatomy_categories import (
    ANATOMY_COLORS,
    GROUPS as ANATOMY_GROUPS,
    OTHER as ANATOMY_OTHER,
)

ROOT = Path(__file__).resolve().parent
UMAP_DIR = ROOT / "outputs" / "concept_umap"
BACKGROUND = "other"  # the unlabelled concept majority, drawn light grey

# The 9 clinical groups spotlighted on the right (fixed clinical order; with
# --order purity these get re-sorted by p, large->small). Row-major layout.
SPOTLIGHT_ORDER = [
    "Abdominal trauma", "Pulmonary embolism", "Pleural effusion",
    "Atherosclerosis", "Bronchiectasis/airway", "Lymphadenopathy",
    "Aortic aneurysm/dissect", "Emphysema", "Pneumothorax",
]

# Original finding-family colors used only by the right spotlight panels.
# Panel A uses the separate, fixed RadLex anatomy palette imported above.
CATEGORY_COLORS = {
    # --- spotlighted (match the rendered figure) ---
    "Abdominal trauma":        "#E41A1C",  # red
    "Pulmonary embolism":      "#1FC3DC",  # cyan
    "Pleural effusion":        "#6A8EDB",  # periwinkle blue
    "Atherosclerosis":         "#4D4D4D",  # charcoal
    "Bronchiectasis/airway":   "#F4A23B",  # orange
    "Lymphadenopathy":         "#8C564B",  # brown
    "Aortic aneurysm/dissect": "#A98FD0",  # lavender
    "Emphysema":               "#B07C74",  # dusty rose-brown
    "Pneumothorax":            "#26318C",  # navy
    # --- left-panel only: distinct, white-visible fillers, one hue family each
    #     (retuned so no two categories share a near-identical colour) ---
    "Nodule/mass":             "#2E7D32",  # forest green
    "Renal":                   "#FA8072",  # salmon (warm: separates from the greens)
    "Ascites/free fluid":      "#8BC34A",  # light green
    "Hepatic":                 "#8E44AD",  # purple
    "Airspace (PNA/GGO)":      "#E7298A",  # magenta
    "Biliary/GB":              "#9E9D24",  # olive
    "Fracture/bone":           "#B8860B",  # goldenrod
    "Interstitial/fibrosis":   "#607D8B",  # slate blue-grey
    "Cardiomegaly/pericardial":"#8B0000",  # dark maroon
}
GREY = "#D9D9D9"


def _load(llm: str, with_spotlight: bool):
    d = UMAP_DIR / llm
    xy = np.load(d / "umap2d.npy")
    anatomy_cats = np.load(d / "radlex_anatomy_categories.npy").astype(str)
    if len(xy) != len(anatomy_cats):
        raise ValueError(
            f"row mismatch: UMAP={len(xy)}, anatomy={len(anatomy_cats)}"
        )
    unknown_anatomy = set(np.unique(anatomy_cats)) - set(ANATOMY_GROUPS)
    if unknown_anatomy:
        raise ValueError(f"unknown anatomical groups: {sorted(unknown_anatomy)}")

    finding_cats = None
    purity = None
    if not with_spotlight:
        return xy, anatomy_cats, finding_cats, purity

    finding_cats = np.load(d / "categories.npy", allow_pickle=True).astype(object)
    purity = json.loads((d / "knn_purity.json").read_text())["per_category"]
    if len(xy) != len(finding_cats):
        raise ValueError(
            f"row mismatch: UMAP={len(xy)}, anatomy={len(anatomy_cats)}, "
            f"findings={len(finding_cats)}"
        )
    unknown_findings = set(np.unique(finding_cats)) - set(CATEGORY_COLORS) - {BACKGROUND}
    if unknown_findings:
        raise ValueError(f"unknown finding groups: {sorted(unknown_findings)}")
    for name in SPOTLIGHT_ORDER:
        actual = int((finding_cats == name).sum())
        expected = int(purity[name]["n"])
        if actual != expected:
            raise ValueError(f"purity/count mismatch for {name}: labels={actual}, purity={expected}")
    return xy, anatomy_cats, finding_cats, purity


# Shared legend styling (also used to pre-measure the legend column width).
# The legend's long labels set the width of the middle gap, so a *big* legend font
# would push the right grid outward. We keep the legend a notch larger than before
# but lean on compact handles (matplotlib reserves a wide handle even for dot markers)
# so the two plots still come closer; the panel TITLES carry the big size bump since
# they sit over the panels and cost no horizontal space.
LEG_FS, LEG_MS, LEG_LABELSPACING, LEG_HANDLEPAD, LEG_HANDLELEN = 12, 11, 0.6, 0.45, 0.7
TITLE_FS = 14


def _legend_width_frac(handles, dpi: int, fig_w_in: float) -> float:
    """Fraction of the figure width the legend occupies at LEG_FS/LEG_MS.

    Independent of where the legend ends up, so we can size the middle column to
    hug the legend exactly (no dead white space) before building the real figure.
    """
    figd = plt.figure(figsize=(fig_w_in, 6), dpi=dpi)
    axd = figd.add_axes([0, 0, 1, 1]); axd.axis("off")
    leg = axd.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.5),
                     frameon=False, fontsize=LEG_FS, labelspacing=LEG_LABELSPACING,
                     handletextpad=LEG_HANDLEPAD, handlelength=LEG_HANDLELEN,
                     borderaxespad=0.0)
    figd.canvas.draw()
    w = leg.get_window_extent(figd.canvas.get_renderer()).width / (fig_w_in * dpi)
    plt.close(figd)
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="f2llm",
                    help="embedding subdir under outputs/concept_umap/ (default: f2llm)")
    ap.add_argument("--out", default=None,
                    help="output image (default: <llm>/concept_latent_anatomy.png)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--spotlight", action="store_true",
                    help="include the optional right-hand 3x3 finding spotlight grid")
    ap.add_argument("--order", choices=["purity", "fixed"], default="purity",
                    help="with --spotlight, order the right grid by kNN purity "
                         "(default) or use the fixed clinical order")
    ap.add_argument("--legend", action=argparse.BooleanOptionalAction, default=True,
                    help="draw the category->colour legend beside the left scatter "
                         "(default: on; use --no-legend to drop it)")
    ap.add_argument("--title", action="store_true",
                    help="add a suptitle (off by default, matching the composite)")
    args = ap.parse_args()

    xy, anatomy_cats, finding_cats, purity = _load(args.llm, args.spotlight)
    out = Path(args.out) if args.out else (UMAP_DIR / args.llm / "concept_latent_anatomy.png")

    # Panel order for the right grid: by purity p descending (per-LLM) or fixed.
    spotlight = list(SPOTLIGHT_ORDER)
    if args.spotlight and args.order == "purity":
        spotlight.sort(key=lambda nm: purity.get(nm, {}).get("purity", float("-inf")),
                       reverse=True)

    # Draw the largest anatomical groups first so small groups remain visible.
    anatomy_draw_order = [g for g in ANATOMY_GROUPS if g != ANATOMY_OTHER and (anatomy_cats == g).any()]
    anatomy_draw_order.sort(key=lambda g: -(anatomy_cats == g).sum())
    is_bg = anatomy_cats == ANATOMY_OTHER

    # Legend handles (biggest clusters first; grey 'other' last) — built up front so
    # the middle column can be sized to fit the enlarged legend snugly.
    # Keep a stable anatomically meaningful legend order (lung -> body wall),
    # independent of prevalence and plotting z-order.
    leg_order = [g for g in ANATOMY_GROUPS if g != ANATOMY_OTHER and (anatomy_cats == g).any()]
    leg_handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=LEG_MS,
                          markerfacecolor=ANATOMY_COLORS[c], markeredgecolor="none",
                          label=c) for c in leg_order]
    if is_bg.any():
        leg_handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=LEG_MS,
                                  markerfacecolor=ANATOMY_COLORS[ANATOMY_OTHER], markeredgecolor="none",
                                  label=ANATOMY_OTHER))

    FIG_H = 8.5
    if args.spotlight:
        FIG_W, L, R = 20.5, 0.01, 0.99
        if args.legend:
            # Size the legend column to hug its text in the optional combined layout.
            frac = _legend_width_frac(leg_handles, args.dpi, FIG_W)
            k = (frac + 0.008) / (R - L)
            leg_w = max(0.18, 2.22 * k / (1.0 - k))
        else:
            leg_w = 0.03
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        outer = fig.add_gridspec(
            1, 3, width_ratios=[1.0, leg_w, 1.22], wspace=0.0,
            left=L, right=R, top=0.94, bottom=0.02,
        )
        legend_slot = outer[0, 1]
    else:
        # A 1:0.44 UMAP-to-legend split gives the square UMAP ~7.4 inches while
        # leaving sufficient room for the longest anatomical legend labels.
        FIG_W = 11.5 if args.legend else 8.5
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        if args.legend:
            outer = fig.add_gridspec(
                1, 2, width_ratios=[1.0, 0.44], wspace=0.01,
                left=0.06, right=0.99, top=0.97, bottom=0.10,
            )
            legend_slot = outer[0, 1]
        else:
            outer = fig.add_gridspec(
                1, 1, left=0.10, right=0.98, top=0.97, bottom=0.10,
            )
            legend_slot = None

    # ---- LEFT: full UMAP coloured by category ------------------------------
    axL = fig.add_subplot(outer[0, 0])
    axL.scatter(xy[is_bg, 0], xy[is_bg, 1], s=1.4, c=GREY, alpha=0.45,
                linewidths=0, rasterized=True)
    for c in anatomy_draw_order:
        m = anatomy_cats == c
        axL.scatter(xy[m, 0], xy[m, 1], s=2.0, c=ANATOMY_COLORS[c], alpha=0.55,
                    linewidths=0, rasterized=True)
    axL.set_box_aspect(1)
    axL.set_xticks([]); axL.set_yticks([])
    axL.set_xlabel("UMAP1", fontsize=18, labelpad=10)
    axL.set_ylabel("UMAP2", fontsize=18, labelpad=10)

    # ---- LEGEND: category -> colour, in the gap before the grid ------------
    if args.legend:
        axLeg = fig.add_subplot(legend_slot); axLeg.axis("off")
        axLeg.legend(handles=leg_handles, loc="center left", bbox_to_anchor=(0.0, 0.5),
                     frameon=False, fontsize=LEG_FS, labelspacing=LEG_LABELSPACING,
                     handletextpad=LEG_HANDLEPAD, handlelength=LEG_HANDLELEN,
                     borderaxespad=0.0)

    # ---- RIGHT: 3x3 spotlight grid -----------------------------------------
    if args.spotlight:
        gridR = outer[0, 2].subgridspec(3, 3, wspace=0.05, hspace=0.30)
        for i, name in enumerate(spotlight):
            ax = fig.add_subplot(gridR[i // 3, i % 3])
            ax.scatter(xy[:, 0], xy[:, 1], s=1.0, c=GREY, alpha=0.35,
                       linewidths=0, rasterized=True)
            m = finding_cats == name
            ax.scatter(xy[m, 0], xy[m, 1], s=2.2,
                       c=CATEGORY_COLORS.get(name, "#333333"), alpha=0.85,
                       linewidths=0, rasterized=True)
            info = purity.get(name, {})
            n, p = info.get("n", int(m.sum())), info.get("purity", float("nan"))
            ax.set_title(f"{name}\n(n={n:,}, p={p:.2f})",
                         fontsize=TITLE_FS, linespacing=1.05)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_edgecolor("black"); s.set_linewidth(1.0)

    if args.title:
        suffix = " · finding-family kNN purity (right)" if args.spotlight else ""
        fig.suptitle(f"Concept latent ({args.llm}): {len(anatomy_cats):,} concepts · "
                     f"RadLex anatomy{suffix}", fontsize=15)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()

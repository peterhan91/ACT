#!/usr/bin/env python3
"""
Combined "headline" figure for exp1 zero-shot concept annotation — the grouped bar
chart and the four circular per-concept AUROC maps merged into the paper layout:

    +-----------------------------+------------------+
    |                             |    CT-RATE       |
    |       grouped bar chart     |   (circular)     |
    |       (Mean AUROC)          |                  |
    +--------------+--------------+------------------+
    |  PMBB chest  | PMBB abdom.  | RSNA Abd. Trauma |
    |  (circular)  | (circular)   |   (circular)     |
    +--------------+--------------+------------------+

The bar chart spans the top-left two columns; the four circular maps fill the
top-right cell and the whole bottom row.

Composition follows stack_panels.py: each piece is rendered to its own PNG and pasted
onto one PIL canvas. The circular panels are fixed-size squares (14in * PANEL_DPI), so
they tile exactly; the bar chart is scaled to fit the 2x1 top-left box and centered.

Two tweaks vs. the standalone figures:
  1. CT-RATE shows the true PATIENT count (config.PATIENT_COUNTS -> 973), not the
     scan/reconstruction count (n_volumes = 2,489 scans). Handled in circular_map.py.
  2. The circular disease/concept labels are enlarged (CONCEPT_LABEL_SIZE * FONT_SCALE).

Reads   analysis/tables/per_label_auc__<dataset>.csv  (via plot_dataset / bar render)
Writes  analysis/plots/figures/combined_figure.png

Run:  python analysis/plots/combined_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # analysis/
import config
from circular_map import plot_dataset
import bar_plot_refstyle as barfig

# ── layout ─────────────────────────────────────────────────────────────────────
TOP_RIGHT  = "ctrate_test"                                       # top-right cell
BOTTOM_ROW = ["pmbb_chest_nc", "pmbb_abd_ce", "rsna2023_test"]   # bottom row, left->right

# ── styling knobs ───────────────────────────────────────────────────────────────
PANEL_DPI        = 200      # circular panel side = 14in * dpi (200 -> 2800px square)
# circle pinned inside each square panel: small enough that the enlarged radial disease
# labels clear the panel edges, big enough that the 7 model-ring labels don't crowd.
PANEL_AX_BOX     = (0.27, 0.27, 0.46, 0.46)   # left, bottom, width, height
MODEL_LABEL_SIZE = 15       # the 7 ring names (Ours..OpenAI CLIP): kept small so they
                            # fit the ring spacing without stacking on top of each other
TITLE_SIZE       = 36       # dataset title (decoupled from the disease-label size)
BAR_FIGSIZE      = (13.5, 6.6)   # ~2:1 so it fills the 2-column top-left box

# Outer disease-label size PER PANEL. Sparse maps (9-18 concepts) get large labels; the
# dense 30-concept PMBB-abdominal map uses a smaller size so adjacent radial labels don't
# collide. Sizes are absolute points (font_scale=1.0), tuned for PANEL_DPI / PANEL_AX_BOX.
CONCEPT_LABEL_SIZE = {
    "ctrate_test":   28,
    "pmbb_chest_nc": 28,
    "pmbb_abd_ce":   18,    # 30 concepts -> smaller to avoid neighbor overlap
    "rsna2023_test": 30,    # only 9 concepts -> plenty of room
}
DEFAULT_CONCEPT_SIZE = 26

PANEL_DIR = config.FIGURES_DIR / "_combined_panels"   # panel renders for THIS figure only
OUT = config.FIGURES_DIR / "combined_figure.png"


def _circular(dataset: str) -> Image.Image:
    p = plot_dataset(dataset, out_dir=PANEL_DIR, colorbar=False,
                     concept_label_size=CONCEPT_LABEL_SIZE.get(dataset, DEFAULT_CONCEPT_SIZE),
                     model_label_size=MODEL_LABEL_SIZE, title_size=TITLE_SIZE,
                     font_scale=1.0, ax_box=PANEL_AX_BOX, dpi=PANEL_DPI)
    return Image.open(p).convert("RGB")


def _fit_centered(img: Image.Image, box_w: int, box_h: int):
    """Scale img to fit inside the box (preserve aspect); return (resized, x_off, y_off)."""
    s = min(box_w / img.width, box_h / img.height)
    new = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))),
                     Image.LANCZOS)
    return new, (box_w - new.width) // 2, (box_h - new.height) // 2


def main() -> None:
    # circular panels (all identical S x S squares)
    ct = _circular(TOP_RIGHT)
    bottom = [_circular(d) for d in BOTTOM_ROW]
    assert ct.width == ct.height, "circular panels must be square"
    S = ct.width

    # bar chart: render near the target 2S width so it stays crisp, then fit & center
    bar_dpi = max(PANEL_DPI, round(2 * S / BAR_FIGSIZE[0]))
    bar_png = barfig.render(out_path=PANEL_DIR / "bar.png", figsize=BAR_FIGSIZE, dpi=bar_dpi)
    bar = Image.open(bar_png).convert("RGB")
    bar_fit, bx, by = _fit_centered(bar, 2 * S, S)

    canvas = Image.new("RGB", (3 * S, 2 * S), "white")
    canvas.paste(bar_fit, (bx, by))        # top-left, spanning 2 columns
    canvas.paste(ct, (2 * S, 0))           # top-right
    for i, im in enumerate(bottom):        # bottom row
        canvas.paste(im, (i * S, S))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"wrote {OUT}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()

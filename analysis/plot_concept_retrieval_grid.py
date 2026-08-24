#!/usr/bin/env python3
"""Re-label the exp3 concept-retrieval hero grid (panel "a") with larger, two-line
row labels.

The CT montage itself (rows = findings, columns = top-10 retrieved CTs, each shown
by its most-attentive axial slice) is the curated figure rendered earlier and stored
as `data/clip3d-concept-retrieval/concept_retrieval_grid_orig.png`. We keep every
image tile pixel-for-pixel and only redraw the left-margin labels: bigger, bold, and
wrapped onto two lines for multi-word findings so the larger font still fits the row.

Source : data/clip3d-concept-retrieval/concept_retrieval_grid_orig.png  (pristine)
Output : data/clip3d-concept-retrieval/concept_retrieval_grid.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "clip3d-concept-retrieval"
SRC = DATA / "concept_retrieval_grid_orig.png"
OUT = DATA / "concept_retrieval_grid.png"

# Row labels, top -> bottom, in the order they appear in the montage.
LABELS = ["Pleural effusion", "Consolidation", "Anasarca", "Atelectasis",
          "Bowel obstruction"]

DPI = 300
MARGIN_PX = 380          # left margin (px) reserved for the labels
PANEL_FONTSIZE = 34      # the "a" panel letter
ROW_FILL = 0.96          # labels may use this fraction of a row's height / margin


def wrap2(label: str) -> str:
    """Split a multi-word label near its middle onto two balanced lines."""
    words = label.split()
    if len(words) < 2:
        return label
    best_i, best_diff = 1, 10 ** 9
    for i in range(1, len(words)):
        diff = abs(len(" ".join(words[:i])) - len(" ".join(words[i:])))
        if diff < best_diff:
            best_diff, best_i = diff, i
    return "\n".join([" ".join(words[:best_i]), " ".join(words[best_i:])])


def detect_grid_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (x0, y0, x1, y1) of the image montage (excludes label margin)."""
    dark = gray < 200
    cols = np.where(dark.mean(axis=0) > 0.5)[0]
    rows = np.where(dark.mean(axis=1) > 0.5)[0]
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def main() -> None:
    src = Image.open(SRC).convert("RGB")
    gray = np.asarray(src.convert("L"))
    x0, y0, x1, y1 = detect_grid_bbox(gray)
    pad = 4
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(src.width, x1 + pad), min(src.height, y1 + pad)
    grid = src.crop((x0, y0, x1, y1))
    gw, gh = grid.size
    n = len(LABELS)

    total_w = gw + MARGIN_PX
    fig = plt.figure(figsize=(total_w / DPI, gh / DPI), dpi=DPI)
    ax = fig.add_axes([MARGIN_PX / total_w, 0.0, gw / total_w, 1.0])
    ax.imshow(grid)
    ax.axis("off")

    row_px = gh / n
    label_x = (MARGIN_PX * 0.5) / total_w

    # First pass: place labels at a trial size, measure, then scale to fit the row.
    texts = []
    for r, lab in enumerate(LABELS):
        y_center = 1.0 - (r + 0.5) / n
        texts.append(fig.text(label_x, y_center, wrap2(lab),
                              fontsize=40, fontweight="bold",
                              ha="center", va="center", rotation=90,
                              linespacing=1.0))
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    max_h = max(t.get_window_extent(rnd).height for t in texts)   # along-row extent
    max_w = max(t.get_window_extent(rnd).width for t in texts)    # margin-direction
    scale = min(ROW_FILL * row_px / max_h, ROW_FILL * MARGIN_PX / max_w)
    label_fs = 40 * scale
    for t in texts:
        t.set_fontsize(label_fs)

    fig.text(0.004, 0.995, "a", fontsize=PANEL_FONTSIZE, fontweight="bold",
             ha="left", va="top")

    fig.savefig(OUT, dpi=DPI)
    plt.close(fig)
    print(f"wrote {OUT}  (label fontsize={label_fs:.1f}pt, grid {gw}x{gh})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Horizontally stack circular-map panels into one multi-panel figure.

The per-dataset circular maps are all identical-size (4200x4200) with the circle pinned
to the same box, so a simple side-by-side composite aligns them perfectly.

  python plots/stack_panels.py                       # default 4: CT-RATE, PMBB chest/abd, RSNA
  python plots/stack_panels.py ctrate_test radchest  # custom order/subset

Assumes the panels were already rendered (plots/make_circular_maps.py).
Writes  analysis/plots/figures/circular_stack.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # analysis/
import config
from circular_map import plot_dataset

PANELS = ["ctrate_test", "pmbb_chest_nc", "pmbb_abd_ce", "rsna2023_test"]
PANEL_FONT_SCALE = 1.5  # enlarge labels/titles for the stacked figure (each panel is 1/N wide)
PANEL_AX_BOX = (0.23, 0.23, 0.54, 0.54)  # smaller CENTERED circle: title + bottom labels both fit
PANEL_DIR = config.FIGURES_DIR / "_panels"   # panel-specific renders (not the standalone figs)
TITLE_SIZE = 42         # one-line subplot title: largest that still fits the panel width
                        # (so it doesn't span the whole 4200² canvas and block the side-crop)
# Each panel is a fixed 4200² canvas with a CENTERED circle, so most of the left/right is white
# margin → big visual gaps when tiled. Crop each panel to its actual ink (circle+labels+title)
# plus a small symmetric pad, so the only horizontal space left between panels is 2·CROP_PAD.
CROP_PAD = 55           # px of white kept on each side of a panel's content (sets the inter-panel gap)


def _hcrop(im: Image.Image, pad: int) -> Image.Image:
    """Trim left/right whitespace to the content bbox (+pad), keeping full height so the
    circles stay vertically aligned. Symmetric content → title stays centred over the circle."""
    g = np.asarray(im.convert("L"))
    cols = np.where((g < 245).any(axis=0))[0]      # columns containing any non-white pixel
    if len(cols) == 0:
        return im
    l = max(0, int(cols[0]) - pad)
    r = min(im.width, int(cols[-1]) + 1 + pad)
    return im.crop((l, 0, r, im.height))


def main(argv: list[str]) -> None:
    panels = argv or PANELS
    imgs = []
    for d in panels:
        # render a panel-specific copy with enlarged fonts (leaves figures/circular_map__*.png alone)
        p = plot_dataset(d, out_dir=PANEL_DIR, colorbar=False,
                         font_scale=PANEL_FONT_SCALE, ax_box=PANEL_AX_BOX, title_size=TITLE_SIZE)
        imgs.append(_hcrop(Image.open(p).convert("RGB"), CROP_PAD))

    h = max(im.height for im in imgs)
    total_w = sum(im.width for im in imgs)
    canvas = Image.new("RGB", (total_w, h), "white")
    x = 0
    for im in imgs:
        canvas.paste(im, (x, (h - im.height) // 2))
        x += im.width

    out = config.FIGURES_DIR / "circular_stack.png"
    canvas.save(out)
    print(f"wrote {out}  ({canvas.size[0]}x{canvas.size[1]}, {len(imgs)} panels: {', '.join(panels)})")


if __name__ == "__main__":
    main(sys.argv[1:])

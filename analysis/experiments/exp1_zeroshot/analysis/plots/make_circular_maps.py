#!/usr/bin/env python3
"""
Render the circular concept-annotation map (PNG) for every exp1 dataset, or a subset.

  python analysis/plots/make_circular_maps.py                 # all datasets
  python analysis/plots/make_circular_maps.py ctrate_test     # one/several

Assumes analysis/aggregate.py has been run (reads analysis/tables/*.csv).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # analysis/
import config
from circular_map import plot_dataset

# ── enlarged text for the standalone single-panel maps ───────────────────────────
# Each standalone map is a full 14x14in / 4200px figure, so it can carry much larger
# labels than the multi-panel combined figure. The circle is pinned smaller (AX_BOX) to
# leave margin for the bigger radial disease labels; the disease-label size is PER-DATASET
# so the dense 30-concept abdominal map doesn't overlap. config.FONT_OVERRIDES (if set for
# a dataset) still wins. model_label_size is capped by the 7-ring radial spacing, so it is
# necessarily more modest than the disease labels / title.
AX_BOX              = (0.28, 0.28, 0.44, 0.44)   # left, bottom, width, height
MODEL_LABEL_SIZE    = 17
# hub/category abbreviations point INWARD toward the centre and converge there, so with the
# smaller circle they must stay small to avoid colliding in the middle (16pt was clean at the
# old big circle; ~12pt is the equivalent here). Everything else can be large; these can't.
CATEGORY_LABEL_SIZE = 12
TITLE_SIZE          = 44
CONCEPT_LABEL_SIZE = {
    "ctrate_test":   31,
    "pmbb_chest_nc": 31,
    "radchest":      31,
    "rsna2023_test": 33,    # only 9 concepts -> plenty of room
    "pmbb_abd_ce":   21,    # 30 concepts -> smaller to avoid neighbor overlap
    "abdominal_combined": 19,
}
DEFAULT_CONCEPT_SIZE = 29


def _sizes(dataset: str) -> dict:
    return dict(
        concept_label_size=CONCEPT_LABEL_SIZE.get(dataset, DEFAULT_CONCEPT_SIZE),
        category_label_size=CATEGORY_LABEL_SIZE,
        model_label_size=MODEL_LABEL_SIZE,
        title_size=TITLE_SIZE,
        ax_box=AX_BOX,
    )


def main(argv: list[str]) -> None:
    known = list(config.DATASETS) + list(getattr(config, "COMPOSITES", {}))
    datasets = argv or known
    unknown = [d for d in datasets if d not in known]
    if unknown:
        raise SystemExit(f"unknown dataset(s): {unknown}\nknown: {known}")
    cbar_only = getattr(config, "COLORBAR_DATASET", None)
    overrides = getattr(config, "FONT_OVERRIDES", {})
    for d in datasets:
        show_cbar = (cbar_only is not None) and (d == cbar_only)
        kw = _sizes(d)
        kw.update(overrides.get(d, {}))   # per-dataset config.FONT_OVERRIDES wins if set
        out = plot_dataset(d, colorbar=show_cbar, **kw)
        print(f"{d:14s} -> {out}  (colorbar={show_cbar})")


if __name__ == "__main__":
    main(sys.argv[1:])

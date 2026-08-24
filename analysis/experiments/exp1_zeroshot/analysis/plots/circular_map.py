"""
Circular per-concept AUROC map (pycirclize) — one figure per dataset.

Concentric rings = models, ordered OUTER->INNER by the CT-RATE ranking
(tables/model_ranking.csv) so the model order is identical across every dataset.
Wedges = concepts grouped into anatomical sectors (concept_groups.py); cell color =
AUROC (seaborn "vlag", 0..1). Mirrors the CXR `plots/circular_map.ipynb` style.

Reads   analysis/tables/per_label_auc__<dataset>.csv  (+ model_ranking.csv)
Writes  analysis/plots/figures/circular_map__<dataset>.png

Use via make_circular_maps.py, or import plot_dataset("ctrate_test").
Requires: pip install pycirclize pandas matplotlib seaborn
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from pycirclize import Circos

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # analysis/
import config
import concept_groups as cg

AUC_CMAP = ListedColormap(sns.color_palette("vlag", as_cmap=False, n_colors=256))

# seaborn sets savefig.bbox='tight' on import, which crops every save to its own content
# (variable size). Restore full-canvas saving so all figures come out identical.
plt.rcParams["savefig.bbox"] = None

# Fixed canvas + fixed circle box so EVERY figure is the same pixel size and the circle
# is the same size/position. AX_BOX pins the polar axes to a centered square, leaving a
# uniform margin for the outer labels; the full canvas is saved (no tight-bbox crop).
FIGSIZE = (14, 14)
# circle pinned to a fixed CENTERED box so the title clears the top labels AND the bottom
# labels clear the bottom edge (equal margin all around).
AX_BOX = (0.18, 0.18, 0.64, 0.64)    # left, bottom, width, height (figure fraction)
TITLE_Y = 0.965


def _wrap2(text: str) -> str:
    """Break a multi-word label into 2 length-balanced lines (1-word labels unchanged)."""
    words = text.split()
    if len(words) < 2:
        return text
    best_i = min(range(1, len(words)),
                 key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
    return " ".join(words[:best_i]) + "\n" + " ".join(words[best_i:])


def _n_volumes(dataset: str):
    """Number of scored volumes/scans for a dataset (from ours' result JSON)."""
    try:
        path = config.result_json("ours", dataset, config.MODELS["ours"]["mode"])
        return json.loads(path.read_text())["n_volumes"]
    except Exception:
        return None


def _n_patients(dataset: str):
    """Patient count for the title: config.PATIENT_COUNTS override if defined, else
    n_volumes (which equals scans for datasets whose scored unit is a volume)."""
    override = getattr(config, "PATIENT_COUNTS", {}).get(dataset)
    return override if override is not None else _n_volumes(dataset)


def _model_order(df: pd.DataFrame) -> list[tuple[str, str]]:
    """[(model_dir, display)] in plot order, keeping only models scored here.

    Uses config.PLOT_ORDER if set (explicit, identical across datasets); otherwise
    falls back to the CT-RATE ranking in model_ranking.csv.
    """
    if getattr(config, "PLOT_ORDER", None):
        order = list(config.PLOT_ORDER)
    else:
        order = pd.read_csv(config.TABLES_DIR / "model_ranking.csv").model.tolist()
    present = [m for m in order
               if f"auc_{m}" in df.columns and df[f"auc_{m}"].notna().any()]
    return [(m, config.MODELS[m]["display"]) for m in present]


def plot_dataset(dataset: str, out_dir: Path | None = None, dpi: int = 300,
                 concept_label_size: int = 16, category_label_size: int = 16,
                 model_label_size: int = 15, colorbar: bool = True,
                 ours_edge_lw: float = 3.0, ours_cell_lw: float = 1.2,
                 font_scale: float = 1.0, title_size: int | None = None,
                 ax_box=None) -> Path:
    composites = getattr(config, "COMPOSITES", {})
    is_composite = dataset in composites
    spec = composites[dataset] if is_composite else config.DATASETS[dataset]
    df = pd.read_csv(config.TABLES_DIR / f"per_label_auc__{dataset}.csv")
    has_ds_band = is_composite and "dataset" in df.columns and df["dataset"].nunique() > 1

    # font_scale enlarges the OUTER concept labels, model legend and title for stacked-panel
    # renders. Category labels are deliberately NOT scaled: they sit in the crowded hub and
    # converge toward the centre, so enlarging them makes them overlap.
    concept_label_size = int(round(concept_label_size * font_scale))
    model_label_size = int(round(model_label_size * font_scale))
    # title defaults to the scaled size, but can be set explicitly (decoupled from the
    # legend/labels) so a panel can have big disease labels and a modestly-sized title.
    title_size = int(round(22 * font_scale)) if title_size is None else int(title_size)
    box = ax_box if ax_box is not None else AX_BOX

    models = _model_order(df)
    n = len(models)
    # sector order follows the (pre-sorted) CSV row order = categories ranked by AUROC
    groups = list(dict.fromkeys(df["group"].tolist()))
    sectors = {g: int((df["group"] == g).sum()) for g in groups}

    circos = Circos(sectors, space=6, start=55, end=360, endspace=False)

    # radius layout: model rings outer->inner; then a category band; for composites an
    # extra dataset band sits between the rings and the category band.
    top, gap, group_h, ds_h = 100.0, 2.0, 3.0, 3.0
    floor = 50.0 if has_ds_band else 45.0
    band = (top - floor) / n
    ring = lambda i: (top - (i + 1) * band, top - i * band)          # (low, high)
    inner = floor
    ds_band = None
    if has_ds_band:
        ds_band = (inner - gap - ds_h, inner - gap)
        inner = ds_band[0]
    group_band = (inner - gap - group_h, inner - gap)

    # global, consistent color per category (same category -> same color across panels)
    cat_colors = {g: cg.category_color(g) for g in groups}

    for sector in circos.sectors:
        sub = df[df["group"] == sector.name].reset_index(drop=True)
        for i, (m, _disp) in enumerate(models):
            vals = sub[f"auc_{m}"].fillna(0.5).tolist()   # NaN -> neutral white
            tr = sector.add_track(ring(i), r_pad_ratio=0.1)
            # our model's ring gets black cell borders (vertical dividers between
            # concepts); baselines keep thin white separators
            ec, lw = ("black", ours_cell_lw) if m == "ours" else ("w", 0.3)
            tr.heatmap([vals], cmap=AUC_CMAP, vmin=0.0, vmax=1.0,
                       rect_kws={"edgecolor": ec, "linewidth": lw})
            # bold black frame around our model's ring to make it stand out
            if m == "ours":
                tr.axis(ec="black", lw=ours_edge_lw)
            elif i == 0:
                tr.axis()
        # concept labels on the outer ring (>=2-word names wrapped to 2 lines)
        outer = sector.tracks[0]
        outer.xticks([i + 0.5 for i in range(len(sub))],
                     labels=[_wrap2(x) for x in sub["display"].tolist()], outer=True,
                     tick_length=0, label_margin=2, label_size=concept_label_size,
                     label_orientation="vertical")
        # group-color ring + sector label
        gtr = sector.add_track(group_band, r_pad_ratio=0.05)
        gtr.heatmap([[1] * len(sub)], cmap=ListedColormap([cat_colors[sector.name]]),
                    vmin=0, vmax=1, rect_kws={"linewidth": 0.0})
        gtr.xticks([(sector.start + sector.end) / 2],
                   labels=[cg.short_category(sector.name)],
                   outer=False, label_size=category_label_size, label_margin=4,
                   label_orientation="vertical")
        # dataset color band (composite only) marking which dataset this sector is from
        if has_ds_band:
            ds_of = sub["dataset"].iloc[0]
            dtr = sector.add_track(ds_band, r_pad_ratio=0.0)
            dtr.heatmap([[1] * len(sub)],
                        cmap=ListedColormap([config.DATASET_BAND[ds_of]["color"]]),
                        vmin=0, vmax=1, rect_kws={"linewidth": 0.0})

    # model legend (ring labels) in the start gap, at each ring's mid-radius
    for i, (_m, disp) in enumerate(models):
        circos.text(" " + disp, r=top - (i + 0.5) * band, color="black",
                    ha="left", va="center", size=model_label_size)

    # dataset legend (composite only): colored names matching the dataset bands
    if has_ds_band:
        for j, m in enumerate(spec["members"]):
            db = config.DATASET_BAND[m]
            nm = _n_volumes(m)
            lab = db["display"] + (f"  (N = {nm:,})" if nm else "")
            circos.text("  " + lab, r=group_band[1] - 1 - j * 6, color=db["color"],
                        ha="left", va="center", size=model_label_size, weight="bold")

    if colorbar:
        # vertical bar beside the model names, in the wedge opening (clear of the rings).
        # "AUROC" is added as the axes TITLE after plotfig (a vertical-colorbar label with
        # rotation=0 sits ON the bar instead of above it).
        circos.colorbar(bounds=(0.74, 0.70, 0.016, 0.15), vmin=0, vmax=1, cmap=AUC_CMAP,
                        orientation="vertical", tick_kws=dict(labelsize=10, colors="black"))

    fig = circos.plotfig(dpi=dpi, figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    # Pin the circle to a fixed box so every figure has an identical-size circle, and
    # disable tight_layout so the box sticks. Outer labels live in the surrounding margin.
    fig.set_layout_engine("none")
    fig.axes[0].set_position(box)
    # Let outer concept labels extend past the axes box instead of being clipped.
    for ax in fig.axes:
        for t in [*ax.texts, ax.xaxis.label, ax.yaxis.label, ax.title]:
            t.set_clip_on(False)

    if colorbar:  # "AUROC" cleanly above the colorbar (smallest axes = the colorbar)
        cbar_ax = min(fig.axes, key=lambda a: a.get_position().width * a.get_position().height)
        cbar_ax.set_title("AUROC", fontsize=12, pad=6)

    if is_composite:
        title = spec["title"]            # per-dataset N shown in the dataset legend
    else:
        n_pat = _n_patients(dataset)
        title = spec["title"] + (f"  (N = {n_pat:,} patients)" if n_pat else "")
    fig.suptitle(title, fontsize=title_size, y=TITLE_Y)

    out_dir = out_dir or config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"circular_map__{dataset}.png"
    fig.savefig(out, dpi=dpi, facecolor="white")  # full fixed canvas -> identical size
    plt.close(fig)
    return out

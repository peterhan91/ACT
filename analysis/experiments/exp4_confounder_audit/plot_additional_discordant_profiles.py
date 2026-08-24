#!/usr/bin/env python3
"""Plot the revised Supplementary Figures 5 and 6 audit galleries.

Each panel shows the unmodified natural positive ranks 1--5 from the exact
Adam-20 all-221 top-25 export.  Bars are mean global probe--observation
projections, whiskers are +/-1 s.d. across probe fits, and open circles are the
twenty fits. All panels use the same red encoding and one shared alignment
axis, so bar lengths compare directly between phenotypes; the more nuanced
interpretation categories remain in the manifest and manuscript prose.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "text.color": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_SELECTION = RESULTS / "additional_discordant_profiles_selection.json"
DEFAULT_PDF = RESULTS / "additional_discordant_profiles.pdf"
DEFAULT_PNG = RESULTS / "additional_discordant_profiles.png"
DEFAULT_CSV = RESULTS / "additional_discordant_profiles_source.csv"

BAR_COLOR = "#CB8A8A"
BAR_EDGE = "#262626"
GRID_COLOR = "#DEDEDE"
EXPECTED_SEEDS = 20
PAGE_SIDE_INCHES = 12.0
GRID_COLUMNS = 2
GRID_ROWS_PER_PAGE = 2
PANELS_PER_FIGURE = GRID_COLUMNS * GRID_ROWS_PER_PAGE
FIRST_SUPPLEMENTARY_FIGURE = 5
# The eight panels moved from Supplementary Figs. 5 and 6 into one main figure.
MAIN_FIGURE_NUMBER = 5
PANEL_AXIS_LEFT = 0.47
PANEL_AXIS_BOTTOM = 0.14
PANEL_AXIS_WIDTH = 0.45
PANEL_AXIS_HEIGHT = 0.70
EXPECTED_THEME_ORDER = [
    "vascular_calcification_atherosclerosis",
] * 3 + [
    "lymph_node_context",
    "effusion_fluid_context",
] + [
    "pulmonary_airspace_disease",
] * 4
VALID_CATEGORIES = {
    "clearly_discordant",
    "plausible_nonspecific_context",
}


def wrap_text(value: str, width: int) -> str:
    value = value[0].upper() + value[1:] if value else value
    return "\n".join(
        textwrap.wrap(
            value,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def validate_selection(data: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = data.get("metadata", {})
    if metadata.get("schema_version") != "additional-discordant-profiles-v5":
        raise ValueError("unexpected selection-manifest schema")
    if metadata.get("selection_status") != "provisional post-hoc illustrative panel set":
        raise ValueError("selection status is not the expected bounded provisional status")
    validation = metadata.get("validation", {})
    required_checks = [
        "nine_panels_labelled_a_to_i",
        "no_overlap_with_figure5_or_supplementary_figure4",
        "nine_distinct_three_digit_phecode_families",
        "three_vascular_one_lymph_one_fluid_four_airspace",
        "recurring_theme_panels_pass_auroc_gate",
        "theme_is_seed_specific_top25_majority_in_all_20_fits",
        "all_displayed_top5_projections_positive_in_all_20_fits",
    ]
    failed = [name for name in required_checks if validation.get(name) is not True]
    if failed:
        raise ValueError(f"selection validation failed or missing: {failed}")

    top25_source = metadata.get("source_files", {}).get("top25_export", {})
    if int(top25_source.get("size_bytes", 0)) <= 0 or not top25_source.get("sha256"):
        raise ValueError("selection manifest lacks a fixed top-25 source identity")

    panels = data.get("panels", [])
    if len(panels) != 9:
        raise ValueError(f"expected nine panels, found {len(panels)}")
    if [panel.get("panel") for panel in panels] != list("abcdefghi"):
        raise ValueError("panel labels must be a through i in order")
    if [panel.get("theme_group") for panel in panels] != EXPECTED_THEME_ORDER:
        raise ValueError(
            "unexpected Figure 5 panel order"
        )

    for panel in panels:
        if panel.get("review_status") != (
            "provisional; independent clinical adjudication not performed"
        ):
            raise ValueError(f"unexpected review status: {panel['phenotype']}")
        category = panel.get("interpretation_category")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"unknown interpretation category: {panel['phenotype']}")
        if (
            panel.get("selection_origin") == "post_hoc_recurring_theme_screen"
            and float(panel["auroc"]["mean"]) < 0.75
        ):
            raise ValueError(f"AUROC gate failed: {panel['phenotype']}")
        if panel.get("seed_specific_top25_stability"):
            stability = panel.get("seed_specific_top25_stability") or {}
            if int(stability.get("minimum", -1)) < 13:
                raise ValueError(
                    f"fit-specific theme stability failed: {panel['phenotype']}"
                )
        if int(panel["auroc"]["n_seeds"]) != EXPECTED_SEEDS:
            raise ValueError(f"expected twenty fits: {panel['phenotype']}")
        rows = panel.get("natural_top25", [])
        if len(rows) != 25 or [row["natural_rank"] for row in rows] != list(
            range(1, 26)
        ):
            raise ValueError(f"malformed natural top 25: {panel['phenotype']}")
        regex = re.compile(panel["theme_regex"], re.IGNORECASE)
        matches = [bool(regex.search(row["concept"])) for row in rows]
        if matches != panel["theme_match_by_natural_rank"]:
            raise ValueError(f"stored theme matches changed: {panel['phenotype']}")
        if sum(matches) != int(panel["theme_count_mean_top25"]):
            raise ValueError(f"stored theme count changed: {panel['phenotype']}")
        for row in rows[:5]:
            if len(row["coefficient"]["per_seed"]) != EXPECTED_SEEDS:
                raise ValueError(
                    f"malformed seed coefficients: {panel['phenotype']}, "
                    f"rank {row['natural_rank']}"
                )
    return panels


def uniform_xmax(panels: list[dict[str, Any]]) -> float:
    """Return the single alignment limit shared by every panel.

    A per-panel limit rescales each axis to its own bars, which makes a 5-unit
    profile and a 19-unit profile look alike. One limit covering the widest
    whisker and the most extreme fit keeps the nine profiles comparable.
    """
    largest = max(
        max(
            row["coefficient"]["mean"] + row["coefficient"]["std"],
            *row["coefficient"]["per_seed"],
        )
        for panel in panels
        for row in panel["natural_top25"][:5]
    )
    raw = largest * 1.03
    step = 1 if raw <= 10 else 2 if raw <= 20 else 5
    return math.ceil(raw / step) * step


def draw_bar(
    axis,
    panel: dict[str, Any],
    rows: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    xmax: float,
    show_xlabel: bool,
    title_x: float = 0.5,
) -> None:
    """Draw the shared square-panel bar, labels, points, and title."""
    means = np.asarray([row["coefficient"]["mean"] for row in rows])
    stds = np.asarray([row["coefficient"]["std"] for row in rows])
    # The two-page square-panel layout leaves enough vertical room to wrap long
    # concepts more tightly and keep their final manuscript size readable.
    wrapped_labels = [wrap_text(row["concept"], width=28) for row in rows]
    line_counts = np.asarray([label.count("\n") + 1 for label in wrapped_labels])

    # Retain the shared audit-figure bar grammar while keeping concept text
    # legible in the square 2-by-2 page layout.
    label_size = 12.3
    label_line_spacing = 0.88
    line_height = label_size * label_line_spacing / 72.0
    block_heights = line_counts * line_height
    axes_height = axis.figure.get_figheight() * axis.get_position().height
    edge_padding = 0.08
    gap = (axes_height - block_heights.sum() - 2 * edge_padding) / (len(rows) - 1)
    if gap < 0.06:
        raise ValueError(
            f"{panel['phenotype']}: labels do not fit in the square-panel geometry; "
            f"computed gap={gap:.3f} in"
        )
    y_positions = []
    cursor = edge_padding
    for block_height in block_heights:
        y_positions.append(cursor + block_height / 2)
        cursor += block_height + gap
    y = np.asarray(y_positions)

    axis.set_axisbelow(True)
    axis.xaxis.grid(True, color="0.87", linewidth=1.15)
    bar_colors = [
        row.get("display_bar_color", panel.get("bar_color", BAR_COLOR))
        for row in rows
    ]
    axis.barh(
        y,
        means,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.0,
        height=0.28,
        zorder=2,
    )
    axis.errorbar(
        means,
        y,
        xerr=stds,
        fmt="none",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=4,
    )
    for yi, row in zip(y, rows):
        values = np.asarray(row["coefficient"]["per_seed"], dtype=float)
        jitter = rng.uniform(-0.07, 0.07, len(values))
        axis.scatter(
            values,
            yi + jitter,
            s=17,
            facecolors="none",
            edgecolors="0.15",
            linewidths=0.75,
            zorder=5,
        )

    axis.axvline(0, color="black", linewidth=1.5)
    axis.set_yticks(y)
    axis.set_yticklabels(
        wrapped_labels,
        fontsize=label_size,
        linespacing=label_line_spacing,
    )
    axis.set_ylim(axes_height, 0)
    axis.tick_params(axis="y", length=4, width=1.0, pad=6)
    axis.tick_params(axis="x", labelsize=12.0)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.set_xlim(-0.035 * xmax, xmax)
    if show_xlabel:
        axis.set_xlabel("Probe-observation alignment", fontsize=14.0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    auc = panel["auroc"]
    lo, hi = auc["ci95_mean"]
    axis.text(
        title_x,
        1.105,
        panel["display"],
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=15.5,
        fontweight="normal",
    )
    axis.text(
        title_x,
        1.02,
        f"(AUROC {auc['mean']:.3f}, 95% CI {lo:.4f} to {hi:.4f})",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=13.3,
        fontweight="normal",
    )


def draw_square_panel_grid(
    figure,
    panels: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    rows: int,
    panels_per_figure: int = PANELS_PER_FIGURE,
    columns: int = GRID_COLUMNS,
) -> None:
    """Draw panels in square cells, restarting labels once per figure.

    The eight panels now form a single main figure, so panels_per_figure is 8
    there and the labels run a to h without restarting.
    """
    # A merged grid may leave trailing cells empty, so allow a short last row.
    if not columns * (rows - 1) < len(panels) <= columns * rows:
        raise ValueError(
            f"expected at most {columns * rows} panels for a {rows}-row grid, "
            f"found {len(panels)}"
        )

    cell_width = 1.0 / columns
    cell_height = 1.0 / rows
    physical_width = cell_width * figure.get_figwidth()
    physical_height = cell_height * figure.get_figheight()
    if not math.isclose(
        physical_width,
        physical_height,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "complete panel cells, including concept labels and bars, are not 1:1"
        )

    title_x = (0.5 - PANEL_AXIS_LEFT) / PANEL_AXIS_WIDTH
    shared_xmax = uniform_xmax(panels)
    for panel_index, panel in enumerate(panels):
        row, column = divmod(panel_index, columns)
        display_panel = chr(ord("a") + panel_index % panels_per_figure)
        cell_x = column * cell_width
        cell_y = 1.0 - (row + 1) * cell_height
        figure.text(
            cell_x + 0.016 * cell_width,
            cell_y + 0.974 * cell_height,
            display_panel,
            fontsize=24,
            fontweight="bold",
            ha="left",
            va="top",
            color="black",
        )
        bar_axis = figure.add_axes(
            [
                cell_x + PANEL_AXIS_LEFT * cell_width,
                cell_y + PANEL_AXIS_BOTTOM * cell_height,
                PANEL_AXIS_WIDTH * cell_width,
                PANEL_AXIS_HEIGHT * cell_height,
            ]
        )
        draw_bar(
            bar_axis,
            panel,
            panel["natural_top25"][:5],
            rng,
            xmax=shared_xmax,
            show_xlabel=row % GRID_ROWS_PER_PAGE == GRID_ROWS_PER_PAGE - 1,
            title_x=title_x,
        )


def export_source_csv(path: Path, panels: list[dict[str, Any]]) -> None:
    fieldnames = [
        "panel",
        "supplementary_figure",
        "display_panel",
        "phenotype",
        "display",
        "phecode",
        "phecode_family",
        "auroc_mean",
        "auroc_ci95_low",
        "auroc_ci95_high",
        "theme_group",
        "theme_label",
        "theme_regex",
        "theme_count_mean_top25",
        "interpretation_category",
        "discordance_class",
        "selection_origin",
        "rationale",
        "review_status",
        "natural_rank",
        "displayed_in_figure",
        "concept_index",
        "concept",
        "theme_match",
        "coefficient_mean",
        "coefficient_std_across_fits",
        "coefficient_per_seed_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for panel_index, panel in enumerate(panels):
            lo, hi = panel["auroc"]["ci95_mean"]
            supplementary_figure = MAIN_FIGURE_NUMBER
            display_panel = chr(ord("a") + panel_index % len(panels))
            for row, theme_match in zip(
                panel["natural_top25"], panel["theme_match_by_natural_rank"]
            ):
                writer.writerow(
                    {
                        "panel": panel["panel"],
                        "supplementary_figure": supplementary_figure,
                        "display_panel": display_panel,
                        "phenotype": panel["phenotype"],
                        "display": panel["display"],
                        "phecode": panel["phecode"],
                        "phecode_family": panel["phecode_family"],
                        "auroc_mean": panel["auroc"]["mean"],
                        "auroc_ci95_low": lo,
                        "auroc_ci95_high": hi,
                        "theme_group": panel["theme_group"],
                        "theme_label": panel["theme_label"],
                        "theme_regex": panel["theme_regex"],
                        "theme_count_mean_top25": panel[
                            "theme_count_mean_top25"
                        ],
                        "interpretation_category": panel[
                            "interpretation_category"
                        ],
                        "discordance_class": panel["discordance_class"],
                        "selection_origin": panel["selection_origin"],
                        "rationale": panel["rationale"],
                        "review_status": panel["review_status"],
                        "natural_rank": row["natural_rank"],
                        "displayed_in_figure": row["natural_rank"] <= 5,
                        "concept_index": row["concept_index"],
                        "concept": row["concept"],
                        "theme_match": theme_match,
                        "coefficient_mean": row["coefficient"]["mean"],
                        "coefficient_std_across_fits": row["coefficient"]["std"],
                        "coefficient_per_seed_json": json.dumps(
                            row["coefficient"]["per_seed"], separators=(",", ":")
                        ),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--export-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    outputs = [args.output_pdf, args.output_png, args.export_csv]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite generated output without --force: "
            + ", ".join(str(path) for path in existing)
        )

    data = json.loads(args.selection.read_text())
    panels = validate_selection(data)
    if int(data["metadata"]["source_files"]["top25_export"].get("size_bytes", 0)) <= 0:
        raise ValueError("invalid source top-25 file metadata")

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    # All eight panels now form a single main figure, so the PDF is one page of
    # 4 rows by 2 columns with continuous a-to-h labels. The 6-by-6-inch cell
    # size is unchanged, so the panels stay legible.
    save_options = {"facecolor": "white"}
    pdf_rng = np.random.default_rng(0)
    merged_columns = 3
    merged_rows = math.ceil(len(panels) / merged_columns)
    with PdfPages(args.output_pdf) as pdf:
        page_figure = plt.figure(
            figsize=(
                PAGE_SIDE_INCHES / GRID_COLUMNS * merged_columns,
                PAGE_SIDE_INCHES / GRID_ROWS_PER_PAGE * merged_rows,
            ),
            facecolor="white",
        )
        draw_square_panel_grid(
            page_figure,
            panels,
            pdf_rng,
            rows=merged_rows,
            panels_per_figure=len(panels),
            columns=merged_columns,
        )
        pdf.savefig(page_figure, **save_options)
        plt.close(page_figure)

    # The PNG preview mirrors the merged PDF layout exactly.
    preview_figure = plt.figure(
        figsize=(
            PAGE_SIDE_INCHES / GRID_COLUMNS * merged_columns,
            PAGE_SIDE_INCHES / GRID_ROWS_PER_PAGE * merged_rows,
        ),
        facecolor="white",
    )
    draw_square_panel_grid(
        preview_figure,
        panels,
        np.random.default_rng(0),
        rows=merged_rows,
        panels_per_figure=len(panels),
        columns=merged_columns,
    )
    preview_figure.savefig(args.output_png, dpi=240, **save_options)
    plt.close(preview_figure)
    export_source_csv(args.export_csv, panels)
    print(f"wrote {args.output_pdf}")
    print(f"wrote {args.output_png}")
    print(f"wrote {args.export_csv}")


if __name__ == "__main__":
    main()

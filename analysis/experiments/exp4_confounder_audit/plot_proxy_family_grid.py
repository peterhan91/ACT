#!/usr/bin/env python3
"""Plot the four-family candidate proxy gallery used as Figure 5.

The eight panels are grouped by the observation family that dominates the
probe's leading directions. Each row holds two phenotypes that share a family
but come from unrelated three-digit phecode groups, and a schematic circle names
that family. Bars are mean global probe-observation projections, whiskers are
+/-1 s.d. across probe fits, and open circles are the twenty fits.

Score magnitudes are comparable within a phenotype but not across phenotypes,
because probe weights are not normalized, so every panel keeps its own x limit.
Pass --uniform-x only when a single shared scale is wanted for display.

Concepts, coefficients and AUROCs come from the fixed all-221 Adam-20
natural-top-25 export.
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
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        # Arial first: the macOS Helvetica .ttc exposes no separate bold face, so
        # fontweight="bold" is silently ignored when Helvetica resolves first.
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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
SOURCE = RESULTS / "f2llm_adam20_top25_all221.json"
DEFAULT_PDF = RESULTS / "proxy_family_grid.pdf"
DEFAULT_PNG = RESULTS / "proxy_family_grid.png"
DEFAULT_CSV = RESULTS / "proxy_family_grid_source.csv"
DEFAULT_MANIFEST = RESULTS / "proxy_family_grid_selection.json"

BAR_COLOR = "#CB8A8A"
CIRCLE_FILL = "#EDEDED"
CIRCLE_TEXT = "#A93226"
EXPECTED_SEEDS = 20
AUROC_GATE = 0.75
FIG_WIDTH, FIG_HEIGHT = 21.0, 21.5

# Row geometry in figure coordinates. The family circle leads each row from the
# far left, so the reader meets the shared direction before the two profiles.
ROW_HEIGHT = 0.178
ROW_GAP = 0.072
FIRST_ROW_BOTTOM = 0.798
CIRCLE_X = 0.0
CIRCLE_WIDTH = 0.185
# COLUMN_X leaves just enough room for the widest wrapped concept label, which
# is right-aligned at COLUMN_X + LABEL_WIDTH and runs about 0.15 wide.
COLUMN_X = (0.192, 0.527)
LABEL_WIDTH = 0.165
BAR_WIDTH = 0.152

DISPLAY = {
    "Dependence on respirator [Ventilator] or supplemental oxygen":
        "Ventilator or supplemental oxygen dependence",
    "Encephalopathy, not elsewhere classified": "Encephalopathy NEC",
    "Hyperosmolality and/or hypernatremia": "Hypernatremia / hyperosmolality",
}

# One row per observation family. The two phenotypes in a row must come from
# different three-digit phecode groups, so a shared family cannot be an artefact
# of two codes for the same disease.
FAMILIES = [
    {
        "key": "device",
        # The regex admits post-procedural change as well as hardware, so the
        # printed label has to name both. Two of the ten displayed strings in
        # this row describe post-operative change and name no device.
        "label": "Drains, stents and prior surgery",
        "regex": r"drain|catheter|stent|post-operative|post-surgical|lobectomy|"
                 r"choledochojejunostomy|resection",
        "phenotypes": [
            "Secondary malignant neoplasm",
            "Adult failure to thrive",
        ],
    },
    {
        "key": "biliary",
        "label": "Biliary ductal dilatation",
        "regex": r"biliary|bile duct|common duct|\bcbd\b|hepatic duct",
        "phenotypes": [
            "Dependence on respirator [Ventilator] or supplemental oxygen",
            "Encephalopathy, not elsewhere classified",
        ],
    },
    {
        "key": "vascular",
        "label": "Aortic / coronary calcification",
        "regex": r"atheroscler|calcific|calcified|atheroma",
        "phenotypes": [
            "Chronic bronchitis",
            "Diaphragmatic hernia",
        ],
    },
    {
        "key": "airspace",
        "label": "Pulmonary air-space disease",
        "regex": r"ground.?glass|consolidat|infiltrat|pneumonic|density increase",
        "phenotypes": [
            "Hyperosmolality and/or hypernatremia",
            "Other abnormal glucose",
        ],
    },
]

MIN_FAMILY_MATCHES_IN_TOP5 = 3
LABEL_WRAP = 34
# One drain-family direction is a long, highly specific report sentence. It is
# shown in full rather than truncated, so the cap allows six lines.
MAX_LABEL_LINES = 6


def capitalize(value: str) -> str:
    return value[0].upper() + value[1:] if value else value


def wrap_concept(value: str) -> str:
    lines = textwrap.wrap(
        capitalize(value),
        width=LABEL_WRAP,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(lines) > MAX_LABEL_LINES:
        raise ValueError(
            f"concept needs {len(lines)} label lines, more than "
            f"{MAX_LABEL_LINES}: {value!r}"
        )
    return "\n".join(lines)


def load_panels(source: Path) -> list[dict[str, Any]]:
    """Read the fixed export and assemble the eight validated panels."""
    data = json.loads(source.read_text())
    metadata = data.get("metadata", {})
    if metadata.get("schema_version") != "f2llm-adam20-top25-v1":
        raise ValueError("unexpected full-bank top-25 source")
    if (
        metadata.get("optimizer") != "Adam"
        or metadata.get("n_initialization_seeds") != EXPECTED_SEEDS
    ):
        raise ValueError("this figure requires the fixed Adam-20 source")

    by_name = {entry["phenotype"]: entry for entry in data["phenotypes"]}
    if len(by_name) != len(data["phenotypes"]):
        raise ValueError("duplicate phenotype names in the top-25 source")

    rank1_counts: dict[str, int] = {}
    for entry in data["phenotypes"]:
        first = entry["natural_top25"][0]["concept"]
        rank1_counts[first] = rank1_counts.get(first, 0) + 1

    panels: list[dict[str, Any]] = []
    for family in FAMILIES:
        pattern = re.compile(family["regex"], re.IGNORECASE)
        for name in family["phenotypes"]:
            entry = by_name[name]
            rows = entry["natural_top25"][:5]
            auroc = entry["auroc"]
            if auroc["mean"] < AUROC_GATE:
                raise ValueError(f"AUROC gate failed: {name}")
            for row in rows:
                per_seed = row["coefficient"]["per_seed"]
                if len(per_seed) != EXPECTED_SEEDS:
                    raise ValueError(f"expected twenty fits: {name}")
                if min(per_seed) <= 0:
                    raise ValueError(
                        f"displayed projection is not positive in all fits: {name}"
                    )
            matches = [bool(pattern.search(row["concept"])) for row in rows]
            if not matches[0]:
                raise ValueError(
                    f"{name}: rank-1 direction is outside the {family['key']} family"
                )
            if sum(matches) < MIN_FAMILY_MATCHES_IN_TOP5:
                raise ValueError(
                    f"{name}: only {sum(matches)} of 5 leading directions are "
                    f"{family['key']}"
                )
            panels.append(
                {
                    "family_key": family["key"],
                    "family_label": family["label"],
                    "family_regex": family["regex"],
                    "phenotype": name,
                    "display": DISPLAY.get(name, capitalize(name)),
                    "phecode": entry["phecode"],
                    "phecode_family": str(entry["phecode"]).split(".")[0],
                    "auroc": auroc,
                    "family_matches_in_top5": sum(matches),
                    "family_match_by_rank": matches,
                    "rank1_shared_with_n_phenotypes": rank1_counts[
                        rows[0]["concept"]
                    ],
                    "rows": rows,
                }
            )

    if len(panels) != 2 * len(FAMILIES):
        raise ValueError(f"expected eight panels, found {len(panels)}")
    groups = [p["phecode_family"] for p in panels]
    if len(set(groups)) != len(groups):
        raise ValueError(f"three-digit phecode groups are not distinct: {groups}")
    return panels


def panel_xmax(panels: list[dict[str, Any]]) -> float:
    """Round one panel's data up to a limit MaxNLocator can tick cleanly."""
    largest = max(
        max(
            row["coefficient"]["mean"] + row["coefficient"]["std"],
            *row["coefficient"]["per_seed"],
        )
        for panel in panels
        for row in panel["rows"]
    )
    raw = largest * 1.06
    step = 1 if raw <= 10 else 2 if raw <= 20 else 5
    return math.ceil(raw / step) * step


def draw_bar(
    axis,
    panel: dict[str, Any],
    rng: np.random.Generator,
    *,
    xmax: float,
    show_xlabel: bool,
) -> None:
    """Draw one phenotype's five leading directions."""
    rows = panel["rows"]
    means = np.asarray([row["coefficient"]["mean"] for row in rows])
    stds = np.asarray([row["coefficient"]["std"] for row in rows])
    wrapped = [wrap_concept(row["concept"]) for row in rows]
    line_counts = np.asarray([label.count("\n") + 1 for label in wrapped])

    label_size = 13.2
    line_spacing = 0.92
    line_height = label_size * line_spacing / 72.0
    block_heights = line_counts * line_height
    axes_height = axis.figure.get_figheight() * axis.get_position().height
    edge_padding = 0.07
    gap = (axes_height - block_heights.sum() - 2 * edge_padding) / (len(rows) - 1)
    if gap < 0.05:
        raise ValueError(
            f"{panel['phenotype']}: labels do not fit the row geometry; "
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
    axis.barh(
        y,
        means,
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=1.0,
        height=0.26,
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
        axis.scatter(
            values,
            yi + rng.uniform(-0.065, 0.065, len(values)),
            s=16,
            facecolors="none",
            edgecolors="0.15",
            linewidths=0.75,
            zorder=5,
        )

    axis.axvline(0, color="black", linewidth=1.5)
    axis.set_yticks(y)
    axis.set_yticklabels(wrapped, fontsize=label_size, linespacing=line_spacing)
    axis.set_ylim(axes_height, 0)
    axis.tick_params(axis="y", length=4, width=1.0, pad=6)
    axis.tick_params(axis="x", labelsize=17.0)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.set_xlim(-0.035 * xmax, xmax)
    if show_xlabel:
        axis.set_xlabel("Probe-observation alignment", fontsize=21.0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    low, high = panel["auroc"]["ci95_mean"]
    axis.text(
        0.5,
        1.115,
        # Wide enough that every displayed phenotype name stays on one line,
        # which keeps a uniform title height above each row of axes.
        textwrap.fill(panel["display"], 46),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=22.0,
        multialignment="center",
    )
    axis.text(
        0.5,
        1.025,
        f"(AUROC {panel['auroc']['mean']:.3f}, 95% CI {low:.4f} to {high:.4f})",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=17.0,
    )


def draw_family_circle(axis, label: str, aspect: float) -> None:
    """Name the observation family that both panels in this row share."""
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    radius = 0.42
    axis.add_patch(
        Ellipse(
            (0.5, 0.5),
            2 * radius,
            2 * radius * aspect,
            facecolor=CIRCLE_FILL,
            edgecolor="black",
            linewidth=2.0,
            zorder=3,
            clip_on=False,
        )
    )
    text = (
        textwrap.fill(
            label, 17, break_long_words=False, break_on_hyphens=False
        )
        + "\n(candidate proxy\ndirection)"
    )
    axis.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=21.0,
        color=CIRCLE_TEXT,
        fontweight="bold",
        zorder=4,
    )


def draw_figure(panels: list[dict[str, Any]], *, uniform_x: bool):
    figure = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor="white")
    rng = np.random.default_rng(0)
    shared_xmax = panel_xmax(panels) if uniform_x else None
    circle_aspect = (CIRCLE_WIDTH * FIG_WIDTH) / (ROW_HEIGHT * FIG_HEIGHT)
    n_rows = len(FAMILIES)

    for index, panel in enumerate(panels):
        row, column = divmod(index, 2)
        bottom = FIRST_ROW_BOTTOM - row * (ROW_HEIGHT + ROW_GAP)
        if column == 0:
            # One letter per row: a panel is a family, not a single phenotype.
            figure.text(
                CIRCLE_X,
                bottom + ROW_HEIGHT + 0.041,
                chr(ord("a") + row),
                fontsize=27,
                fontweight="bold",
                ha="left",
                va="top",
                color="black",
            )
        axis = figure.add_axes(
            [
                COLUMN_X[column] + LABEL_WIDTH,
                bottom,
                BAR_WIDTH,
                ROW_HEIGHT,
            ]
        )
        draw_bar(
            axis,
            panel,
            rng,
            xmax=shared_xmax or panel_xmax([panel]),
            show_xlabel=row == n_rows - 1,
        )
        if column == 0:
            circle = figure.add_axes(
                [CIRCLE_X, bottom, CIRCLE_WIDTH, ROW_HEIGHT]
            )
            draw_family_circle(circle, panel["family_label"], circle_aspect)
    return figure


def export_source_csv(path: Path, panels: list[dict[str, Any]]) -> None:
    fieldnames = [
        "panel",
        "position",
        "family_key",
        "family_label",
        "family_regex",
        "phenotype",
        "display",
        "phecode",
        "phecode_family",
        "auroc_mean",
        "auroc_ci95_low",
        "auroc_ci95_high",
        "family_matches_in_top5",
        "rank1_shared_with_n_phenotypes",
        "natural_rank",
        "concept_index",
        "concept",
        "family_match",
        "coefficient_mean",
        "coefficient_std_across_fits",
        "coefficient_per_seed_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, panel in enumerate(panels):
            low, high = panel["auroc"]["ci95_mean"]
            for row, match in zip(panel["rows"], panel["family_match_by_rank"]):
                writer.writerow(
                    {
                        "panel": chr(ord("a") + index // 2),
                        "position": "left" if index % 2 == 0 else "right",
                        "family_key": panel["family_key"],
                        "family_label": panel["family_label"],
                        "family_regex": panel["family_regex"],
                        "phenotype": panel["phenotype"],
                        "display": panel["display"],
                        "phecode": panel["phecode"],
                        "phecode_family": panel["phecode_family"],
                        "auroc_mean": panel["auroc"]["mean"],
                        "auroc_ci95_low": low,
                        "auroc_ci95_high": high,
                        "family_matches_in_top5": panel["family_matches_in_top5"],
                        "rank1_shared_with_n_phenotypes": panel[
                            "rank1_shared_with_n_phenotypes"
                        ],
                        "natural_rank": row["natural_rank"],
                        "concept_index": row["concept_index"],
                        "concept": row["concept"],
                        "family_match": match,
                        "coefficient_mean": row["coefficient"]["mean"],
                        "coefficient_std_across_fits": row["coefficient"]["std"],
                        "coefficient_per_seed_json": json.dumps(
                            row["coefficient"]["per_seed"], separators=(",", ":")
                        ),
                    }
                )


def export_manifest(
    path: Path, panels: list[dict[str, Any]], *, uniform_x: bool
) -> None:
    payload = {
        "metadata": {
            "schema_version": "proxy-family-grid-v1",
            "source": str(SOURCE.relative_to(SOURCE.parents[3])),
            "selection_status": "provisional post-hoc illustrative panel set",
            "review_status": "provisional; independent clinical adjudication "
            "not performed",
            "auroc_gate": AUROC_GATE,
            "x_axis": "shared across panels" if uniform_x else "per panel",
            "claim_boundary": "Global probe-observation alignment only; not "
            "evidence that an observation occurred in an individual scan, "
            "caused a prediction, or was used as a patient-level shortcut.",
        },
        "panels": [
            {
                "panel": chr(ord("a") + index // 2),
                "position": "left" if index % 2 == 0 else "right",
                "family_key": panel["family_key"],
                "family_label": panel["family_label"],
                "phenotype": panel["phenotype"],
                "display": panel["display"],
                "phecode": panel["phecode"],
                "phecode_family": panel["phecode_family"],
                "auroc": panel["auroc"],
                "family_matches_in_top5": panel["family_matches_in_top5"],
                "rank1_shared_with_n_phenotypes": panel[
                    "rank1_shared_with_n_phenotypes"
                ],
                "per_fit_stability": panel.get("per_fit_stability"),
                "natural_top5": [
                    {
                        "natural_rank": row["natural_rank"],
                        "concept_index": row["concept_index"],
                        "concept": row["concept"],
                        "coefficient": row["coefficient"],
                    }
                    for row in panel["rows"]
                ],
            }
            for index, panel in enumerate(panels)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--export-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--export-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--uniform-x",
        action="store_true",
        help="share one x limit across panels; magnitudes are not comparable "
        "between phenotypes, so this is a display choice only",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    outputs = [
        args.output_pdf,
        args.output_png,
        args.export_csv,
        args.export_manifest,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite generated output without --force: "
            + ", ".join(str(path) for path in existing)
        )

    panels = load_panels(args.source)

    # Per-fit family stability is recomputed over the whole bank by a separate
    # pass, because it needs the 20 raw weight vectors and the concept bank.
    stability_path = RESULTS / "proxy_family_grid_perfit_stability.json"
    if stability_path.exists():
        stability = {
            entry["phenotype"]: entry
            for entry in json.loads(stability_path.read_text()).values()
        }
        for panel in panels:
            panel["per_fit_stability"] = stability.get(panel["phenotype"])

    for index, panel in enumerate(panels):
        print(
            f"{chr(ord('a') + index)} {panel['display'][:44]:46s} "
            f"phecode {str(panel['phecode']):>7} "
            f"AUROC {panel['auroc']['mean']:.3f} "
            f"{panel['family_key']} {panel['family_matches_in_top5']}/5"
        )

    figure = draw_figure(panels, uniform_x=args.uniform_x)
    save_options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.03}
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_pdf, **save_options)
    figure.savefig(args.output_png, dpi=220, **save_options)
    plt.close(figure)

    export_source_csv(args.export_csv, panels)
    export_manifest(args.export_manifest, panels, uniform_x=args.uniform_x)
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

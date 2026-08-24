#!/usr/bin/env python3
"""Build paired full-bank -> refined supplementary probe-audit figures.

Each phenotype occupies one square cell in a 2-by-2 figure, matching the
publication geometry and visual grammar of Supplementary Figs. 7 and 8.  The
cell contains two horizontal-bar audits:

    Full-bank top 5 (prior)  ->  Refined top 5 (later)

The arrow denotes analytical ordering, not patient time or causal replacement.
Full-bank observations are natural mean ranks 1--5.  Refined observations are
the five largest bootstrap-mean weights among the fitted restricted probe's
20 largest positive point-fit weights.  No de-duplication or clinical filtering
is applied after ranking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
FIGURES = HERE / "figures"
DEFAULT_SELECTION = RESULTS / "refinement_paired_audit_selection.json"
DEFAULT_SOURCE_CSV = RESULTS / "refinement_paired_audit_source.csv"

FULL_COLOR = "#CB8A8A"
REFINED_COLOR = "#5B9BD5"
FULL_TEXT = "#9E3F3F"
REFINED_TEXT = "#2C6FAD"
GRID_COLOR = "#DEDEDE"
EDGE_COLOR = "#262626"
EXPECTED_REPLICATES = 20
PAGE_SIDE_INCHES = 8.0
N_COLUMNS = 2
N_ROWS = 2

GROUPS = {
    "higher": [
        ("Cardiomegaly", "Cardiomegaly"),
        ("Osteoporosis NOS", "Osteoporosis NOS"),
        ("Other aneurysm", "Other aneurysm"),
        (
            "Pulmonary embolism and infarction, acute",
            "Acute pulmonary embolism/infarction",
        ),
    ],
    "lower": [
        ("Diseases of pancreas", "Diseases of pancreas"),
        ("Cancer of bronchus; lung", "Cancer of bronchus/lung"),
        ("Osteoarthrosis", "Osteoarthrosis"),
        ("Diaphragmatic hernia", "Diaphragmatic hernia"),
    ],
}

CURRENT_FIGURE6_PHENOTYPES = {
    "Pneumococcal pneumonia",
    "Congestive heart failure (CHF) NOS",
    "Shock",
    "Hypovolemia",
    "Other pulmonary inflamation or edema",
    "Pneumonitis due to inhalation of food or vomitus",
    "Asphyxia and hypoxemia",
    "Pericarditis",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_text(value: str) -> str:
    """Apply typography-only normalization without changing clinical meaning."""
    value = value.strip()
    value = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " × ", value)
    value = re.sub(r"\btype i\b", "Type I", value, flags=re.IGNORECASE)
    value = re.sub(r"\btype iii\b", "Type III", value, flags=re.IGNORECASE)
    value = re.sub(r"\bge junction\b", "GE junction", value, flags=re.IGNORECASE)
    value = re.sub(r"\bsma\b", "SMA", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\b([ctl])(\d)(?:-([ctl])?(\d))?\b",
        lambda match: (
            f"{match.group(1).upper()}{match.group(2)}"
            + (
                f"–{(match.group(3) or match.group(1)).upper()}{match.group(4)}"
                if match.group(4)
                else ""
            )
        ),
        value,
        flags=re.IGNORECASE,
    )
    return value[0].upper() + value[1:] if value else value


def wrap_text(value: str, width: int = 24) -> str:
    return "\n".join(
        textwrap.wrap(
            display_text(value),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def _compact_full(row: dict[str, Any]) -> dict[str, Any]:
    coefficient = row["coefficient"]
    return {
        "rank": int(row["natural_rank"]),
        "concept_index": int(row["concept_index"]),
        "concept": row["concept"],
        "mean": float(coefficient["mean"]),
        "std": float(coefficient["std"]),
        "replicates": [float(value) for value in coefficient["per_seed"]],
    }


def _compact_refined(row: dict[str, Any], display_rank: int) -> dict[str, Any]:
    return {
        "rank": display_rank,
        "point_rank": int(row["point_rank"]),
        "concept_index": int(row["concept_index"]),
        "concept": row["concept"],
        "tier": row.get("tier"),
        "mean": float(row["importance_boot_mean"]),
        "std": float(row["importance_boot_std"]),
        "replicates": [float(value) for value in row["per_seed"]],
        "positive_seed_fraction": float(row["positive_seed_fraction"]),
        "sign_consistency_with_point": float(row["sign_consistency_with_point"]),
    }


def build_selection(ledger_path: Path) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text())
    rows_by_name = {row["phenotype"]: row for row in ledger["rows"]}
    panels: dict[str, list[dict[str, Any]]] = {}

    for group, specifications in GROUPS.items():
        group_panels = []
        expected_positive = group == "higher"
        for panel_index, (phenotype, display) in enumerate(specifications):
            row = rows_by_name[phenotype]
            delta = float(row["auc_delta_refined_minus_full"])
            if (delta > 0) != expected_positive:
                raise ValueError(f"unexpected AUROC direction for {phenotype}: {delta}")
            full = [_compact_full(item) for item in row["full_top20"][:5]]
            refined = [
                _compact_refined(item, display_rank)
                for display_rank, item in enumerate(
                    row["refined_top20_point_selected_then_mean_ordered"][:5],
                    start=1,
                )
            ]
            group_panels.append(
                {
                    "panel": chr(ord("a") + panel_index),
                    "phenotype": phenotype,
                    "display": display,
                    "phecode": str(row["phecode"]),
                    "full_auc_mean": float(row["full_auc_mean"]),
                    "full_auc_std_across_20_fits": float(
                        row["full_auc_std_across_20_fits"]
                    ),
                    "full_auc_per_fit": [
                        float(value) for value in row["full_auc_per_seed"]
                    ],
                    "refined_auc_point": float(row["refined_auc_point"]),
                    "refined_auc_test_resample_mean": float(
                        row["refined_auc_test_resample_mean"]
                    ),
                    "refined_auc_test_resample_std": float(
                        row["refined_auc_test_resample_std"]
                    ),
                    "auc_delta_refined_minus_full": delta,
                    "n_refined_concepts": int(row["n_refined_concepts"]),
                    "full_top5": full,
                    "refined_top5": refined,
                }
            )
        panels[group] = group_panels

    all_names = {
        panel["phenotype"] for group_panels in panels.values() for panel in group_panels
    }
    overlap = sorted(all_names & CURRENT_FIGURE6_PHENOTYPES)
    if overlap:
        raise ValueError(f"selected phenotypes overlap current Figure 6: {overlap}")

    for group_panels in panels.values():
        for panel in group_panels:
            if len(panel["full_top5"]) != 5 or len(panel["refined_top5"]) != 5:
                raise ValueError(f"expected two top-five profiles: {panel['phenotype']}")
            for condition in ("full_top5", "refined_top5"):
                for row in panel[condition]:
                    if len(row["replicates"]) != EXPECTED_REPLICATES:
                        raise ValueError(
                            f"expected 20 values: {panel['phenotype']} {condition}"
                        )

    return {
        "metadata": {
            "schema_version": "refinement-paired-audit-v1",
            "selection_status": "post-hoc clinically reviewed illustrative profiles",
            "source_ledger": {
                "path": str(ledger_path.resolve()),
                "size_bytes": ledger_path.stat().st_size,
                "sha256": sha256(ledger_path),
                "schema_version": ledger["metadata"]["schema_version"],
            },
            "comparison": (
                "current full-bank mean AUROC across 20 validation-selected fits "
                "versus current fitted refined-probe point AUROC"
            ),
            "full_profile_rule": "natural full-bank mean ranks 1-5",
            "refined_profile_rule": (
                "select the fitted refined probe's 20 largest positive point-fit "
                "weights, then display the five largest bootstrap-mean weights"
            ),
            "n_full_fits": 20,
            "n_refined_training_bootstraps": 20,
            "no_deduplication_or_post_rank_filtering": True,
            "arrow_meaning": (
                "analytical full-bank to refined comparison; not patient time or "
                "causal feature replacement"
            ),
            "claim_boundary": (
                "global probe-observation profiles selected post hoc for manual "
                "clinical review; not patient-level attribution, diagnostic "
                "validation, or an estimate of prevalence"
            ),
            "validation": {
                "four_higher_and_four_lower": True,
                "no_exact_current_figure6_phenotype_overlap": True,
                "all_profiles_have_five_rows_and_twenty_replicates": True,
            },
        },
        "groups": panels,
    }


def export_csv(path: Path, selection: dict[str, Any]) -> None:
    fields = [
        "supplementary_figure",
        "panel",
        "group",
        "phenotype",
        "display",
        "phecode",
        "full_auc_mean",
        "refined_auc_point",
        "auc_delta_refined_minus_full",
        "condition",
        "rank",
        "point_rank",
        "concept_index",
        "concept",
        "tier",
        "score_mean",
        "score_std",
        "replicates_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group, panels in selection["groups"].items():
            supplementary_figure = 7 if group == "higher" else 8
            for panel in panels:
                for condition, key in (
                    ("full", "full_top5"),
                    ("refined", "refined_top5"),
                ):
                    for row in panel[key]:
                        writer.writerow(
                            {
                                "supplementary_figure": supplementary_figure,
                                "panel": panel["panel"],
                                "group": group,
                                "phenotype": panel["phenotype"],
                                "display": panel["display"],
                                "phecode": panel["phecode"],
                                "full_auc_mean": panel["full_auc_mean"],
                                "refined_auc_point": panel["refined_auc_point"],
                                "auc_delta_refined_minus_full": panel[
                                    "auc_delta_refined_minus_full"
                                ],
                                "condition": condition,
                                "rank": row["rank"],
                                "point_rank": row.get("point_rank", ""),
                                "concept_index": row["concept_index"],
                                "concept": row["concept"],
                                "tier": row.get("tier", ""),
                                "score_mean": row["mean"],
                                "score_std": row["std"],
                                "replicates_json": json.dumps(row["replicates"]),
                            }
                        )


def label_layout(
    labels: list[str],
    axes_height_inches: float,
    *,
    initial_size: float = 6.8,
) -> tuple[np.ndarray, float, float]:
    line_counts = np.asarray([label.count("\n") + 1 for label in labels])
    for label_size in np.arange(initial_size, 5.4, -0.2):
        line_spacing = 0.86
        line_height = label_size * line_spacing / 72.0
        blocks = line_counts * line_height
        padding = 0.035
        gap = (axes_height_inches - blocks.sum() - 2 * padding) / 4
        if gap >= 0.025:
            y = []
            cursor = padding
            for block in blocks:
                y.append(cursor + block / 2)
                cursor += block + gap
            return np.asarray(y), float(label_size), line_spacing
    raise ValueError("concept labels do not fit the square paired-audit panel")


def score_limits(rows: list[dict[str, Any]]) -> tuple[float, float]:
    values = np.concatenate(
        [np.asarray(row["replicates"], dtype=float) for row in rows]
    )
    lows = [row["mean"] - row["std"] for row in rows]
    highs = [row["mean"] + row["std"] for row in rows]
    lo = min(float(values.min()), min(lows), 0.0)
    hi = max(float(values.max()), max(highs), 0.0)
    span = max(hi - lo, 1e-6)
    return lo - 0.05 * span, hi + 0.08 * span


def draw_audit(
    figure,
    *,
    label_bounds: list[float],
    bar_bounds: list[float],
    rows: list[dict[str, Any]],
    color: str,
    text_color: str,
    rng: np.random.Generator,
    show_xlabel: bool,
) -> None:
    label_axis = figure.add_axes(label_bounds)
    bar_axis = figure.add_axes(bar_bounds)
    labels = [wrap_text(row["concept"]) for row in rows]
    axes_height = figure.get_figheight() * bar_bounds[3]
    y, label_size, line_spacing = label_layout(labels, axes_height)

    label_axis.set_xlim(0, 1)
    label_axis.set_ylim(axes_height, 0)
    label_axis.axis("off")
    for yi, label in zip(y, labels):
        label_axis.text(
            0.98,
            yi,
            label,
            ha="right",
            va="center",
            fontsize=label_size,
            linespacing=line_spacing,
            color=text_color,
        )

    means = np.asarray([row["mean"] for row in rows])
    stds = np.asarray([row["std"] for row in rows])
    bar_axis.set_axisbelow(True)
    bar_axis.xaxis.grid(True, color=GRID_COLOR, linewidth=0.75)
    bar_axis.barh(
        y,
        means,
        color=color,
        edgecolor=EDGE_COLOR,
        linewidth=0.7,
        height=0.18,
        zorder=2,
    )
    bar_axis.errorbar(
        means,
        y,
        xerr=stds,
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=2.2,
        capthick=0.9,
        zorder=4,
    )
    for yi, row in zip(y, rows):
        values = np.asarray(row["replicates"], dtype=float)
        bar_axis.scatter(
            values,
            yi + rng.uniform(-0.052, 0.052, len(values)),
            s=6.5,
            facecolors="none",
            edgecolors="0.15",
            linewidths=0.45,
            zorder=5,
        )
    bar_axis.axvline(0, color="black", linewidth=0.9)
    bar_axis.set_ylim(axes_height, 0)
    bar_axis.set_yticks([])
    bar_axis.set_xlim(*score_limits(rows))
    bar_axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
    bar_axis.tick_params(axis="x", labelsize=5.9, pad=1.5)
    if show_xlabel:
        bar_axis.set_xlabel("Observation weight", fontsize=6.4, labelpad=1.5)
    for side in ("top", "right", "left"):
        bar_axis.spines[side].set_visible(False)


def draw_figure(
    group: str,
    panels: list[dict[str, Any]],
    pdf_path: Path,
    png_path: Path,
) -> None:
    if len(panels) != 4:
        raise ValueError(f"{group}: expected four panels")
    figure = plt.figure(figsize=(PAGE_SIDE_INCHES, PAGE_SIDE_INCHES))
    rng = np.random.default_rng(17 if group == "higher" else 29)

    cell_width = 0.5
    cell_height = 0.5
    if not math.isclose(
        cell_width * figure.get_figwidth(),
        cell_height * figure.get_figheight(),
        abs_tol=1e-12,
    ):
        raise RuntimeError("paired-audit subplot cells must be square")

    for index, panel in enumerate(panels):
        row, column = divmod(index, 2)
        cell_x = column * cell_width
        cell_y = 1.0 - (row + 1) * cell_height
        panel_letter = chr(ord("a") + index)
        delta = panel["auc_delta_refined_minus_full"]
        delta_text = f"{delta:+.3f}".replace("-", "−")

        figure.text(
            cell_x + 0.017 * cell_width,
            cell_y + 0.972 * cell_height,
            panel_letter,
            fontsize=16.0,
            fontweight="bold",
            ha="left",
            va="top",
        )
        figure.text(
            cell_x + 0.5 * cell_width,
            cell_y + 0.928 * cell_height,
            panel["display"],
            fontsize=10.2,
            ha="center",
            va="top",
        )
        figure.text(
            cell_x + 0.5 * cell_width,
            cell_y + 0.855 * cell_height,
            (
                f"AUROC {panel['full_auc_mean']:.3f} "
                + r"$\rightarrow$"
                + f" {panel['refined_auc_point']:.3f} ({delta_text})"
            ),
            fontsize=8.2,
            ha="center",
            va="top",
        )
        figure.text(
            cell_x + 0.235 * cell_width,
            cell_y + 0.782 * cell_height,
            "Full-bank top 5\n(prior)",
            fontsize=7.1,
            fontweight="bold",
            color=FULL_TEXT,
            ha="center",
            va="top",
            linespacing=0.9,
        )
        figure.text(
            cell_x + 0.5 * cell_width,
            cell_y + 0.745 * cell_height,
            r"$\longrightarrow$",
            fontsize=15.0,
            fontweight="bold",
            ha="center",
            va="center",
        )
        figure.text(
            cell_x + 0.765 * cell_width,
            cell_y + 0.782 * cell_height,
            "Refined top 5\n(later)",
            fontsize=7.1,
            fontweight="bold",
            color=REFINED_TEXT,
            ha="center",
            va="top",
            linespacing=0.9,
        )

        body_y = cell_y + 0.115 * cell_height
        body_height = 0.575 * cell_height
        show_xlabel = row == 1
        draw_audit(
            figure,
            label_bounds=[
                cell_x + 0.015 * cell_width,
                body_y,
                0.285 * cell_width,
                body_height,
            ],
            bar_bounds=[
                cell_x + 0.305 * cell_width,
                body_y,
                0.145 * cell_width,
                body_height,
            ],
            rows=panel["full_top5"],
            color=FULL_COLOR,
            text_color=FULL_TEXT,
            rng=rng,
            show_xlabel=show_xlabel,
        )
        draw_audit(
            figure,
            label_bounds=[
                cell_x + 0.535 * cell_width,
                body_y,
                0.285 * cell_width,
                body_height,
            ],
            bar_bounds=[
                cell_x + 0.825 * cell_width,
                body_y,
                0.16 * cell_width,
                body_height,
            ],
            rows=panel["refined_top5"],
            color=REFINED_COLOR,
            text_color=REFINED_TEXT,
            rng=rng,
            show_xlabel=show_xlabel,
        )

    figure.text(
        0.5,
        0.012,
        (
            "Global probe-observation profiles; separate score axes. "
            "Arrow indicates analytical comparison, not patient time or causality."
        ),
        fontsize=6.4,
        color="0.35",
        ha="center",
        va="bottom",
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches=None, facecolor="white")
    figure.savefig(png_path, dpi=220, bbox_inches=None, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        type=Path,
        help="Current matched 86-phenotype ledger; rebuilds the compact selection.",
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()

    if args.ledger:
        selection = build_selection(args.ledger)
        args.selection.parent.mkdir(parents=True, exist_ok=True)
        args.selection.write_text(json.dumps(selection, indent=2) + "\n")
        export_csv(args.source_csv, selection)
    else:
        selection = json.loads(args.selection.read_text())
        if selection.get("metadata", {}).get("schema_version") != (
            "refinement-paired-audit-v1"
        ):
            raise ValueError("unexpected paired-audit selection schema")

    for group in ("higher", "lower"):
        draw_figure(
            group,
            selection["groups"][group],
            args.output_dir / f"refinement_{group}_auroc_profiles.pdf",
            args.output_dir / f"refinement_{group}_auroc_profiles.png",
        )
    print(f"wrote {args.selection}")
    print(f"wrote {args.source_csv}")
    print(f"wrote paired figures under {args.output_dir}")


if __name__ == "__main__":
    main()

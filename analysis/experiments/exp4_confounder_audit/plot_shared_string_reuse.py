#!/usr/bin/env python3
"""Plot how few distinct observations fill the leading ranks of many phenotypes.

Two observation strings each occupy rank 1 for a large group of phenotypes that
share no clinical target. For each group this figure shows, cell by cell, how
widely the observation occupying that rank is reused inside the group, and how
much the complete top 25 overlaps between pairs of its members.

No alignment magnitude is plotted, so nothing here invites a comparison of score
magnitudes between phenotypes, which unnormalized probe weights do not support.

Concepts and AUROCs come from the fixed all-221 Adam-20 natural-top-25 export.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize


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
DEFAULT_PDF = RESULTS / "shared_string_reuse.pdf"
DEFAULT_PNG = RESULTS / "shared_string_reuse.png"
DEFAULT_CSV = RESULTS / "shared_string_reuse_source.csv"

# Single-hue sequential ramp in the house red, light to dark, anchored on the
# accent red used by the other audit figures. Relative luminance is monotone
# decreasing and the light end sits at 1.33:1 against white, so the near-zero
# step recedes without vanishing.
RED_RAMP = [
    "#efdbd9", "#e9cac7", "#e4b8b3", "#dfa5a0", "#da928b", "#d77f76",
    "#d46b61", "#d1574b", "#cf4234", "#c1372a", "#af2f23", "#9d281c",
    "#8a2116",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("shared_red", RED_RAMP)
SURFACE = "#ffffff"
GRID_INK = "#d6d6d3"
TEXT_MUTED = "#52514e"

TOP_K = 5
FULL_K = 25

VASCULAR = (
    "calcific atherosclerotic changes in the thoracoabdominal aorta "
    "and coronary artery walls"
)
BILIARY = "extensive dilatation of the biliary system and common duct"

DISPLAY = {
    "Persons with potential health hazards related to socioeconomic, "
    "psychosocial, and other circumstances":
        "Socioeconomic and psychosocial circumstances",
    "Dependence on respirator [Ventilator] or supplemental oxygen":
        "Ventilator or supplemental oxygen dependence",
    "Encephalopathy, not elsewhere classified": "Encephalopathy NEC",
    "Delirium dementia and amnestic and other cognitive disorders":
        "Delirium, dementia and cognitive disorders",
    "Delirium due to conditions classified elsewhere":
        "Delirium due to other conditions",
    "Esophagitis, GERD and related diseases": "Esophagitis and GERD",
    "Toxic effect of (non-ethyl) alcohol and petroleum and other solvents":
        "Toxic effect of alcohol and solvents",
    "Other symptoms/disorders or the urinary system":
        "Other urinary system symptoms",
    "Chronic Kidney Disease, Stage III": "Chronic kidney disease, stage III",
}


def capitalize(value: str) -> str:
    return value[0].upper() + value[1:] if value else value


def load_groups(source: Path) -> list[dict[str, Any]]:
    """Collect every phenotype whose rank-1 direction is one of the two strings."""
    data = json.loads(source.read_text())
    if data.get("metadata", {}).get("schema_version") != "f2llm-adam20-top25-v1":
        raise ValueError("unexpected full-bank top-25 source")
    entries = data["phenotypes"]

    groups = []
    for key, string in (("vascular", VASCULAR), ("biliary", BILIARY)):
        members = [e for e in entries if e["natural_top25"][0]["concept"] == string]
        if len(members) < 2:
            raise ValueError(f"{key}: expected a shared rank-1 string")
        members.sort(key=lambda e: -e["auroc"]["mean"])
        top5 = {e["phenotype"]: [r["concept"] for r in e["natural_top25"][:TOP_K]]
                for e in members}
        top25 = {e["phenotype"]: set(r["concept"] for r in e["natural_top25"])
                 for e in members}
        for name, rows in top25.items():
            if len(rows) != FULL_K:
                raise ValueError(f"{name}: malformed top 25")
        # How many members of this group carry each observation in their top 5.
        reuse: dict[str, int] = {}
        for concepts in top5.values():
            for concept in set(concepts):
                reuse[concept] = reuse.get(concept, 0) + 1
        overlaps = [
            len(top25[a["phenotype"]] & top25[b["phenotype"]])
            for a, b in itertools.combinations(members, 2)
        ]
        slots = [c for concepts in top5.values() for c in concepts]
        groups.append(
            {
                "key": key,
                "string": string,
                "members": members,
                "top5": top5,
                "reuse": reuse,
                "overlaps": overlaps,
                "n_slots": len(slots),
                "n_distinct": len(set(slots)),
            }
        )
    return groups


def draw_reuse_matrix(axis, group: dict[str, Any], norm) -> None:
    """One row per phenotype, one column per natural rank."""
    members = group["members"]
    n = len(members)
    matrix = np.zeros((n, TOP_K))
    for i, entry in enumerate(members):
        for j, concept in enumerate(group["top5"][entry["phenotype"]]):
            matrix[i, j] = 100.0 * group["reuse"][concept] / n

    axis.imshow(
        matrix,
        cmap=SEQUENTIAL,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )
    # A surface-coloured gap between fills keeps adjacent cells separable.
    axis.set_xticks(np.arange(-0.5, TOP_K, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, n, 1), minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=2.0)
    axis.tick_params(which="minor", length=0)

    axis.set_xticks(range(TOP_K))
    axis.set_xticklabels([str(r + 1) for r in range(TOP_K)], fontsize=21)
    axis.set_xlabel("Natural rank", fontsize=22)
    axis.xaxis.set_label_position("top")
    axis.xaxis.tick_top()
    axis.set_yticks(range(n))
    axis.set_yticklabels(
        [
            f"{DISPLAY.get(e['phenotype'], capitalize(e['phenotype']))} "
            f"({e['auroc']['mean']:.3f})"
            for e in members
        ],
        fontsize=20,
    )
    axis.tick_params(axis="both", length=0)
    for side in axis.spines.values():
        side.set_visible(False)


def draw_figure(groups: list[dict[str, Any]]):
    """Two reuse matrices stacked, sharing one vertical colour key at the right.

    Stacked rather than side by side because a side-by-side pair is about twice
    as wide as it is tall, and scaling that into one text column would drop the
    row labels to roughly six point. The canvas is proportioned so that at
    \\textwidth the labels land near ten point.
    """
    figure = plt.figure(figsize=(12.5, 18.6), facecolor=SURFACE)
    norm = Normalize(vmin=0, vmax=100)

    row_height = 0.0205
    matrix_left, matrix_width = 0.590, 0.260
    label_offset = 0.566
    string_wrap = 60

    tops = (0.900, 0.360)
    bottoms = []
    for (label, group), top in zip(zip("ab", groups), tops):
        n = len(group["members"])
        height = n * row_height
        bottom = top - height
        bottoms.append(bottom)
        axis = figure.add_axes([matrix_left, bottom, matrix_width, height])
        draw_reuse_matrix(axis, group, norm)

        # Counts and medians live in the caption, not repeated on the page. The
        # figure carries only what the grid cannot say: which string is shared.
        header_left = matrix_left - label_offset
        figure.text(
            header_left, top + 0.096, label,
            fontsize=30, fontweight="bold", ha="left", va="top",
        )
        figure.text(
            header_left + 0.030, top + 0.092,
            textwrap.fill(
                "Shared rank-1 observation: " + capitalize(group["string"]),
                string_wrap,
            ),
            fontsize=22, ha="left", va="top", linespacing=1.35,
        )

    # One vertical key at the right edge, spanning both grids.
    bar = figure.add_axes(
        [matrix_left + matrix_width + 0.055, bottoms[-1], 0.020,
         tops[0] - bottoms[-1]]
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL)
    colorbar = figure.colorbar(mappable, cax=bar, orientation="vertical")
    colorbar.set_label(
        "Phenotypes with this observation in their top 5 (%)", fontsize=22
    )
    colorbar.ax.tick_params(labelsize=19)
    colorbar.outline.set_visible(False)
    return figure


def export_source_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    fieldnames = [
        "panel", "group", "shared_rank1_concept", "n_phenotypes_in_group",
        "n_top5_slots", "n_distinct_concepts_in_top5", "phenotype", "display",
        "phecode", "auroc_mean", "natural_rank", "concept",
        "n_group_phenotypes_with_concept_in_top5",
        "share_of_group_percent",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for panel, group in zip("ab", groups):
            n = len(group["members"])
            for entry in group["members"]:
                name = entry["phenotype"]
                for rank, concept in enumerate(group["top5"][name], start=1):
                    writer.writerow(
                        {
                            "panel": panel,
                            "group": group["key"],
                            "shared_rank1_concept": group["string"],
                            "n_phenotypes_in_group": n,
                            "n_top5_slots": group["n_slots"],
                            "n_distinct_concepts_in_top5": group["n_distinct"],
                            "phenotype": name,
                            "display": DISPLAY.get(name, capitalize(name)),
                            "phecode": entry["phecode"],
                            "auroc_mean": entry["auroc"]["mean"],
                            "natural_rank": rank,
                            "concept": concept,
                            "n_group_phenotypes_with_concept_in_top5":
                                group["reuse"][concept],
                            "share_of_group_percent":
                                round(100.0 * group["reuse"][concept] / n, 2),
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--export-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    outputs = [args.output_pdf, args.output_png, args.export_csv]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite generated output without --force: "
            + ", ".join(str(p) for p in existing)
        )

    groups = load_groups(args.source)
    for panel, group in zip("ab", groups):
        print(
            f"{panel} {group['key']:9s} n={len(group['members']):2d} "
            f"slots={group['n_slots']:3d} distinct={group['n_distinct']:3d} "
            f"overlap min/median/max="
            f"{min(group['overlaps'])}/"
            f"{statistics.median(group['overlaps']):.0f}/"
            f"{max(group['overlaps'])}"
        )

    figure = draw_figure(groups)
    save = {"facecolor": SURFACE, "bbox_inches": "tight", "pad_inches": 0.04}
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_pdf, **save)
    figure.savefig(args.output_png, dpi=200, **save)
    plt.close(figure)
    export_source_csv(args.export_csv, groups)
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

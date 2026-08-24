#!/usr/bin/env python3
"""Plot four clinically reviewed full-bank probe-observation profiles.

The source JSON contains the unmodified natural full-bank top-10 rankings for
all 221 INSPECT phenotypes. The displayed set comprises four distinct phecode
families whose complete natural top 10 was clinically reviewed as either direct
target evidence or clinically related but non-defining/qualified context.

The one-page PDF uses the same square-panel renderer as the additional
discordant-profile figures. It is a 2-by-2 supplementary figure in which each
complete title-label-bar cell is exactly 1:1. The figure displays unmodified
natural ranks 1--5 without
filtering, replacement or reranking. Dark-blue bars denote direct target
evidence and light-blue bars denote clinically related but non-defining or
qualified context. This display-relevance axis is stored separately from the
conservative rule-based retain/reject audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from plot_additional_discordant_profiles import (
    GRID_ROWS_PER_PAGE,
    PAGE_SIDE_INCHES,
    PANELS_PER_FIGURE,
    draw_square_panel_grid,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_SOURCE = RESULTS / "f2llm_20seed_natural_top10_clinical_audit.json"
DEFAULT_PNG = HERE / "figures" / "clinically_aligned_f2llm.png"
DEFAULT_PDF = HERE / "figures" / "clinically_aligned_f2llm.pdf"
DEFAULT_CSV = RESULTS / "clinically_aligned_f2llm_source.csv"
DEFAULT_SELECTION = RESULTS / "clinically_aligned_f2llm_selection.json"

EXPECTED_SEEDS = 20
FIRST_SUPPLEMENTARY_FIGURE = 4
DIRECT_BAR_COLOR = "#2F6DAE"
INDIRECT_BAR_COLOR = "#8FBCE6"
NO_TARGET_SIGNAL_BAR_COLOR = "#D9D9D9"
DIRECT_RELEVANCE = "direct target evidence"
RELATED_RELEVANCE = "clinically related but non-defining or qualified context"
NO_RELEVANCE = "no qualifying clinical relationship"
MAIN_FIGURE5_PHENOTYPES = {
    "Shock",
    "Anemia in neoplastic disease",
    "Secondary malignancy of bone",
    "Diaphragmatic hernia",
    "Lymphadenitis",
    "Other disorders of liver",
}


PANELS: list[dict[str, Any]] = [
    {
        "phenotype": "Bronchiectasis",
        "display": "Bronchiectasis",
        "expected_retained": 10,
        "allowed_retained_decisions": {"retain_direct"},
        "expected_top5_reasons": {"direct_ct_finding": 5},
        "interpretation_class": "strict direct profile",
        "interpretation": (
            "All ten natural observations were retained as direct bronchiectatic "
            "morphology."
        ),
        "reviewed_relevance_by_rank": [DIRECT_RELEVANCE] * 10,
        "reviewed_relevance_rationale": (
            "All ten phrases directly describe bronchiectatic morphology."
        ),
    },
    {
        "phenotype": "Pneumonia",
        "display": "Pneumonia",
        "expected_retained": 9,
        "allowed_retained_decisions": {"retain_imaging_support"},
        "expected_top5_reasons": {
            "uncertain_language": 1,
            "direct_imaging_pattern_not_etiology": 4,
        },
        "interpretation_class": "qualified imaging-support profile",
        "interpretation": (
            "Nine natural observations were retained as imaging support rather than "
            "etiologic proof; one uncertain observation was rejected."
        ),
        "reviewed_relevance_by_rank": [RELATED_RELEVANCE] * 10,
        "reviewed_relevance_rationale": (
            "All ten phrases describe air-space disease that can support pneumonia "
            "but is not etiologically specific; uncertainty wording limits direct-tier "
            "eligibility without removing clinical relatedness."
        ),
    },
    {
        "phenotype": "Coronary atherosclerosis",
        "display": "Coronary atherosclerosis",
        "expected_retained": 2,
        "allowed_retained_decisions": {"retain_direct"},
        "expected_top5_reasons": {
            "target_mismatch": 3,
            "direct_ct_finding": 2,
        },
        "interpretation_class": "partial direct profile",
        "interpretation": (
            "Two natural observations mentioning coronary calcification were retained; "
            "aortic-only observations provide systemic atherosclerotic context rather "
            "than direct coronary evidence."
        ),
        "reviewed_relevance_by_rank": [
            RELATED_RELEVANCE,
            RELATED_RELEVANCE,
            RELATED_RELEVANCE,
            DIRECT_RELEVANCE,
            DIRECT_RELEVANCE,
            RELATED_RELEVANCE,
            RELATED_RELEVANCE,
            RELATED_RELEVANCE,
            RELATED_RELEVANCE,
            RELATED_RELEVANCE,
        ],
        "reviewed_relevance_rationale": (
            "Coronary-containing phrases are direct; aortic-only phrases are related "
            "systemic atherosclerotic context but do not establish coronary disease."
        ),
    },
    {
        "phenotype": "Pulmonary collapse; interstitial and compensatory emphysema",
        "display": "Pulmonary collapse; interstitial\nand compensatory emphysema",
        "expected_retained": 6,
        "allowed_retained_decisions": {"retain_branch_specific"},
        "expected_top5_reasons": {
            "temporal_comparison": 1,
            "direct_composite_branch": 3,
            "uncertain_language": 1,
        },
        "interpretation_class": "composite-branch profile",
        "interpretation": (
            "Six natural observations were retained for the collapse branch of the "
            "composite phenotype."
        ),
        "reviewed_relevance_by_rank": [RELATED_RELEVANCE] * 10,
        "reviewed_relevance_rationale": (
            "All ten phrases describe atelectasis/collapse-branch findings; temporal "
            "or uncertain wording makes them qualified rather than direct."
        ),
    },
]


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retained_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in entry["audited_top10"] if row["audit"]["retain"]]
    return sorted(rows, key=lambda row: row["natural_rank"])


def display_relevance_style(relevance: str) -> str:
    """Map the separate clinical display-review axis to a publication colour."""
    color_by_relevance = {
        DIRECT_RELEVANCE: DIRECT_BAR_COLOR,
        RELATED_RELEVANCE: INDIRECT_BAR_COLOR,
        NO_RELEVANCE: NO_TARGET_SIGNAL_BAR_COLOR,
    }
    try:
        return color_by_relevance[relevance]
    except KeyError as error:
        raise ValueError(f"unmapped display relevance: {relevance}") from error


def validate_selection_rule(
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    source_names = {entry["phenotype"] for entry in entries}
    selected_names = [panel["phenotype"] for panel in PANELS]
    if len(selected_names) != 4 or len(set(selected_names)) != 4:
        raise ValueError("expected four unique selected phenotypes")
    missing = sorted(set(selected_names) - source_names)
    if missing:
        raise ValueError(f"selected phenotypes missing from source: {missing}")
    overlap = sorted(set(selected_names) & MAIN_FIGURE5_PHENOTYPES)
    if overlap:
        raise ValueError(
            "supplementary clinical panels overlap Main Figure 5: "
            f"{overlap}"
        )

    by_name = {entry["phenotype"]: entry for entry in entries}
    phecode_families = {
        str(by_name[name]["phecode"]).split(".", maxsplit=1)[0]
        for name in selected_names
    }
    if len(phecode_families) != 4:
        raise ValueError(
            "expected four distinct three-digit phecode families, found "
            f"{sorted(phecode_families)}"
        )

    reviewed_classes: list[str] = []
    for panel in PANELS:
        relevance = panel["reviewed_relevance_by_rank"]
        if len(relevance) != 10:
            raise ValueError(
                f"{panel['phenotype']}: expected ten reviewed relevance classes"
            )
        reviewed_classes.extend(relevance)
    unexpected = sorted(
        set(reviewed_classes) - {DIRECT_RELEVANCE, RELATED_RELEVANCE}
    )
    if unexpected:
        raise ValueError(
            "selected top-10 rows include a non-blue display class: "
            f"{unexpected}"
        )

    return {
        "four_unique_phenotypes": True,
        "four_distinct_three_digit_phecode_families": True,
        "no_overlap_with_main_figure_5_phenotypes": True,
        "all_selected_natural_top10_rows_clinically_related": True,
        "strict_audit_fields_preserved_as_separate_axis": True,
        "reviewed_natural_top10_relevance_class_counts": dict(
            Counter(reviewed_classes)
        ),
    }


def validate_and_select(
    data: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = data["metadata"]
    if metadata["concept_bank_size"] != 376_194:
        raise ValueError("expected the 376,194-observation full-bank audit")
    if metadata["probe"] != "Adam linear probe, lr=3e-2, 20 initialization seeds":
        raise ValueError("unexpected probe source")
    if not all(metadata["provenance_checks"].values()):
        raise ValueError("the retained audit source failed a provenance check")

    entries = data["phenotypes"]
    if len(entries) != 221:
        raise ValueError(f"expected 221 phenotype profiles, found {len(entries)}")
    validation = validate_selection_rule(entries)
    by_name = {entry["phenotype"]: entry for entry in entries}

    selected: list[dict[str, Any]] = []
    for panel_index, definition in enumerate(PANELS):
        entry = by_name[definition["phenotype"]]
        retained = retained_rows(entry)
        if len(retained) != definition["expected_retained"]:
            raise ValueError(
                f"{definition['phenotype']}: expected "
                f"{definition['expected_retained']} retained, found {len(retained)}"
            )
        decisions = {row["audit"]["decision"] for row in retained}
        if not decisions <= definition["allowed_retained_decisions"]:
            raise ValueError(
                f"{definition['phenotype']}: unexpected retained decisions {decisions}"
            )
        if int(entry["auroc"]["n_seeds"]) != EXPECTED_SEEDS:
            raise ValueError(
                f"{definition['phenotype']}: expected {EXPECTED_SEEDS} AUROC fits"
            )

        natural_top10 = entry["natural_top10"]
        if len(natural_top10) != 10 or [
            row["natural_rank"] for row in natural_top10
        ] != list(range(1, 11)):
            raise ValueError(f"{definition['phenotype']}: malformed natural top 10")
        top5_reasons = Counter(
            row["audit"]["reason_code"] for row in natural_top10[:5]
        )
        if top5_reasons != Counter(definition["expected_top5_reasons"]):
            raise ValueError(
                f"{definition['phenotype']}: displayed audit outcomes changed: "
                f"{dict(top5_reasons)}"
            )

        plot_rows: list[dict[str, Any]] = []
        for row, display_relevance in zip(
            natural_top10,
            definition["reviewed_relevance_by_rank"],
            strict=True,
        ):
            if len(row["coefficient"]["per_seed"]) != EXPECTED_SEEDS:
                raise ValueError(
                    f"{definition['phenotype']}, rank {row['natural_rank']}: "
                    f"expected {EXPECTED_SEEDS} coefficients"
                )
            bar_color = display_relevance_style(display_relevance)
            plot_rows.append(
                {
                    **row,
                    "display_relevance_class": display_relevance,
                    "display_relevance_rationale": definition[
                        "reviewed_relevance_rationale"
                    ],
                    "display_bar_color": bar_color,
                }
            )

        selected.append(
            {
                **definition,
                "canonical_panel": chr(ord("a") + panel_index),
                "supplementary_figure": (
                    FIRST_SUPPLEMENTARY_FIGURE
                    + panel_index // PANELS_PER_FIGURE
                ),
                "display_panel": chr(
                    ord("a") + panel_index % PANELS_PER_FIGURE
                ),
                "phecode": str(entry["phecode"]),
                "auroc": entry["auroc"],
                "audit_summary": entry["audit_summary"],
                # The shared square renderer consumes this generic row field.
                "natural_top25": plot_rows,
            }
        )
    return selected, validation


def export_source_csv(
    path: Path,
    panels: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "canonical_panel",
        "supplementary_figure",
        "display_panel",
        "phenotype",
        "display",
        "phecode",
        "auroc_mean",
        "auroc_ci95_low",
        "auroc_ci95_high",
        "interpretation_class",
        "interpretation",
        "natural_top10_retained",
        "natural_rank",
        "displayed_in_figure",
        "concept_index",
        "concept",
        "audit_classifier_tier",
        "audit_classifier_score",
        "audit_decision",
        "audit_retain",
        "audit_reason_code",
        "audit_reason",
        "audit_evidence_scope",
        "display_relevance_class",
        "display_relevance_rationale",
        "display_bar_color",
        "coefficient_mean",
        "coefficient_std_across_fits",
        "coefficient_per_seed_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for panel in panels:
            lo, hi = panel["auroc"]["ci95_mean"]
            for row in panel["natural_top25"]:
                audit = row["audit"]
                writer.writerow(
                    {
                        "canonical_panel": panel["canonical_panel"],
                        "supplementary_figure": panel["supplementary_figure"],
                        "display_panel": panel["display_panel"],
                        "phenotype": panel["phenotype"],
                        "display": panel["display"],
                        "phecode": panel["phecode"],
                        "auroc_mean": panel["auroc"]["mean"],
                        "auroc_ci95_low": lo,
                        "auroc_ci95_high": hi,
                        "interpretation_class": panel["interpretation_class"],
                        "interpretation": panel["interpretation"],
                        "natural_top10_retained": panel["audit_summary"]["n_retained"],
                        "natural_rank": row["natural_rank"],
                        "displayed_in_figure": row["natural_rank"] <= 5,
                        "concept_index": row["concept_index"],
                        "concept": row["concept"],
                        "audit_classifier_tier": audit["classifier_tier"],
                        "audit_classifier_score": audit["classifier_score"],
                        "audit_decision": audit["decision"],
                        "audit_retain": audit["retain"],
                        "audit_reason_code": audit["reason_code"],
                        "audit_reason": audit["reason"],
                        "audit_evidence_scope": audit["evidence_scope"],
                        "display_relevance_class": row[
                            "display_relevance_class"
                        ],
                        "display_relevance_rationale": row[
                            "display_relevance_rationale"
                        ],
                        "display_bar_color": row["display_bar_color"],
                        "coefficient_mean": row["coefficient"]["mean"],
                        "coefficient_std_across_fits": row["coefficient"]["std"],
                        "coefficient_per_seed_json": json.dumps(
                            row["coefficient"]["per_seed"], separators=(",", ":")
                        ),
                    }
                )


def export_selection_json(
    path: Path,
    source: Path,
    panels: list[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    output = {
        "metadata": {
            "schema_version": "clinical-alignment-audit-panels-v5",
            "generated_on": date.today().isoformat(),
            "selection_rule": [
                "Select four clinically reviewed full-bank profiles from four distinct three-digit phecode families.",
                "Require every unmodified natural top-10 observation in each selected profile to be direct target evidence or clinically related but non-defining or qualified context.",
                "Exclude phenotypes shown in Main Figure 5 and retain the natural ranking without filtering, replacement or reranking.",
            ],
            "display": (
                "Unmodified natural ranks 1-5; dark-blue bars denote direct target "
                "evidence and light-blue bars denote clinically related but "
                "non-defining or qualified context. Display relevance is separate "
                "from the conservative strict-audit retain/reject decision."
            ),
            "color_key": {
                DIRECT_RELEVANCE: DIRECT_BAR_COLOR,
                RELATED_RELEVANCE: INDIRECT_BAR_COLOR,
                NO_RELEVANCE: NO_TARGET_SIGNAL_BAR_COLOR,
            },
            "source": {
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": sha256(source),
            },
            "validation": validation,
            "clinical_review": "independent clinical adjudication not performed",
        },
        "panels": [
            {
                "canonical_panel": panel["canonical_panel"],
                "supplementary_figure": panel["supplementary_figure"],
                "display_panel": panel["display_panel"],
                "phenotype": panel["phenotype"],
                "display": panel["display"],
                "phecode": panel["phecode"],
                "auroc": panel["auroc"],
                "interpretation_class": panel["interpretation_class"],
                "interpretation": panel["interpretation"],
                "reviewed_relevance_rationale": panel[
                    "reviewed_relevance_rationale"
                ],
                "audit_summary": panel["audit_summary"],
                "displayed_top5_reason_code_counts": dict(
                    Counter(
                        row["audit"]["reason_code"]
                        for row in panel["natural_top25"][:5]
                    )
                ),
                "displayed_top5_relevance_class_counts": dict(
                    Counter(
                        row["display_relevance_class"]
                        for row in panel["natural_top25"][:5]
                    )
                ),
                "reviewed_top10_relevance_class_counts": dict(
                    Counter(
                        row["display_relevance_class"]
                        for row in panel["natural_top25"]
                    )
                ),
            }
            for panel in panels
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--export-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--export-selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    outputs = [
        args.output_pdf,
        args.output_png,
        args.export_csv,
        args.export_selection,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite generated output without --force: "
            + ", ".join(str(path) for path in existing)
        )

    data = json.loads(args.source.read_text())
    panels, validation = validate_and_select(data)

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"facecolor": "white"}

    pdf_rng = np.random.default_rng(0)
    with PdfPages(args.output_pdf) as pdf:
        for page_start in range(0, len(panels), PANELS_PER_FIGURE):
            page_figure = plt.figure(
                figsize=(PAGE_SIDE_INCHES, PAGE_SIDE_INCHES),
                facecolor="white",
            )
            draw_square_panel_grid(
                page_figure,
                panels[page_start : page_start + PANELS_PER_FIGURE],
                pdf_rng,
                rows=GRID_ROWS_PER_PAGE,
            )
            pdf.savefig(page_figure, **save_options)
            plt.close(page_figure)

    n_pages = (len(panels) + PANELS_PER_FIGURE - 1) // PANELS_PER_FIGURE
    preview_figure = plt.figure(
        figsize=(PAGE_SIDE_INCHES, PAGE_SIDE_INCHES * n_pages),
        facecolor="white",
    )
    draw_square_panel_grid(
        preview_figure,
        panels,
        np.random.default_rng(0),
        rows=GRID_ROWS_PER_PAGE * n_pages,
    )
    preview_figure.savefig(args.output_png, dpi=240, **save_options)
    plt.close(preview_figure)

    export_source_csv(args.export_csv, panels)
    export_selection_json(
        args.export_selection,
        args.source,
        panels,
        validation,
    )
    print(f"wrote {args.output_pdf}")
    print(f"wrote {args.output_png}")
    print(f"wrote {args.export_csv}")
    print(f"wrote {args.export_selection}")


if __name__ == "__main__":
    main()

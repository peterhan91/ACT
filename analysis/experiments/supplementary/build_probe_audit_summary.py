#!/usr/bin/env python3
"""Build the manuscript-facing summary of the deterministic probe audit.

This script intentionally summarizes the rule screen itself.  It does not
estimate the prevalence of shortcuts, causal proxy use, or patient-level
explanation fidelity.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SOURCE = (
    REPO
    / "experiments"
    / "exp4_confounder_audit"
    / "results"
    / "f2llm_20seed_natural_top10_clinical_audit.json"
)
DEFAULT_OUT = HERE / "outputs" / "probe_audit"


def _escape_tex(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    return "".join(replacements.get(char, char) for char in value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = json.loads(args.source.read_text())
    summary = payload["inspection_summary"]
    phenotypes = payload["phenotypes"]

    n_phenotypes = int(summary["n_phenotypes"])
    n_audited = int(summary["n_natural_concepts_audited"])
    n_retained = int(summary["n_audit_retained_concepts"])
    decisions = summary["retained_decision_counts"]

    retained_per_phenotype = [
        int(item["audit_summary"]["n_retained"]) for item in phenotypes
    ]
    phenotype_none = sum(value == 0 for value in retained_per_phenotype)
    phenotype_partial = sum(0 < value < 10 for value in retained_per_phenotype)
    phenotype_all = sum(value == 10 for value in retained_per_phenotype)

    assert len(phenotypes) == n_phenotypes == 221
    assert n_audited == n_phenotypes * 10 == 2210
    assert sum(int(value) for value in decisions.values()) == n_retained == 30
    assert phenotype_none + phenotype_partial + phenotype_all == n_phenotypes
    assert (phenotype_none, phenotype_partial, phenotype_all) == (215, 5, 1)

    rows = [
        ("Observation-level", "Top-ranked observations screened", n_audited),
        ("Observation-level", "Retained as direct evidence", int(decisions["retain_direct"])),
        (
            "Observation-level",
            "Retained as branch-specific evidence",
            int(decisions["retain_branch_specific"]),
        ),
        (
            "Observation-level",
            "Retained as imaging-support evidence",
            int(decisions["retain_imaging_support"]),
        ),
        ("Observation-level", "Not retained by the screen", n_audited - n_retained),
        ("Phenotype-level", "No retained observation among the top 10", phenotype_none),
        ("Phenotype-level", "One to nine retained observations", phenotype_partial),
        ("Phenotype-level", "All 10 observations retained", phenotype_all),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "probe_audit_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["analysis_level", "screen_result", "count"])
        writer.writerows(rows)

    tex_path = args.out_dir / "probe_audit_summary.tex"
    tex_lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\normalsize",
        r"\caption{\textbf{Deterministic screen of the ten highest-ranked observations for each INSPECT phenotype.} The screen evaluated 2,210 observation--phenotype pairs from 221 probes. Counts characterize this author-defined keyword and regular-expression screen, not the prevalence of shortcuts or patient-level use of an observation.}",
        r"\label{tab:probe_audit_summary}",
        r"\begin{tabularx}{\textwidth}{@{}lXr@{}}",
        r"\toprule",
        r"\textbf{Analysis level} & \textbf{Screen result} & \textbf{Count} \\",
        r"\midrule",
    ]
    for analysis_level, screen_result, count in rows:
        tex_lines.append(
            f"{_escape_tex(analysis_level)} & {_escape_tex(screen_result)} & {count:,} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    tex_path.write_text("\n".join(tex_lines))

    status = {
        "source": str(args.source.resolve()),
        "outputs": [str(csv_path.resolve()), str(tex_path.resolve())],
        "validated_counts": {
            "phenotypes": n_phenotypes,
            "pairs": n_audited,
            "retained": n_retained,
            "phenotypes_none_partial_all": [
                phenotype_none,
                phenotype_partial,
                phenotype_all,
            ],
        },
        "claim_boundary": (
            "Rule-screen summary only; not shortcut prevalence, causal evidence, "
            "or patient-level explanation validation."
        ),
    }
    (args.out_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(tex_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Matched restriction sensitivity analysis for the 86 INSPECT phenotypes.

This script compares fixed test-set predictions from the two named profiles:

* ``original top-100 profile``
* ``rule-restricted profile``

It uses one shared patient-cluster bootstrap for both profiles and all 86
phenotypes. Each bootstrap replicate samples INSPECT ``person_id`` clusters
with replacement and gives every volume from a sampled patient the same
multiplicity. Point AUROCs are evaluated over all test volumes; percentile
confidence intervals describe test-cohort sampling variability at the patient
cluster level. The script does not retrain either model and does not interpret
the restriction rules as clinical validation.

Outputs are written only below ``experiments/supplementary/outputs`` by
default. No person identifier is written to any output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score


ORIGINAL_PROFILE = "original top-100 profile"
RESTRICTED_PROFILE = "rule-restricted profile"
BOOTSTRAP_METHOD = (
    "paired patient-cluster percentile bootstrap with shared patient resamples"
)
DISPLAY_PHENOTYPE_CORRECTIONS = {
    # Correct a spelling error in the retained concept metadata for
    # publication-facing outputs; alignment is validated against the original
    # label before this display-only normalization is applied.
    "Other pulmonary inflamation or edema": "Other pulmonary inflammation or edema",
}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRUSTED_PROBE_DIR = (
    ROOT / "outputs/v1/external/trusted_concept_probe_f2llm_lbfgs"
)
DEFAULT_OUTPUT = HERE / "outputs/matched_restriction_sensitivity"


@dataclass(frozen=True)
class LoadedData:
    phenotype_records: list[dict[str, Any]]
    phenotype_names: list[str]
    phecodes: list[str]
    volumes: np.ndarray
    person_ids: np.ndarray
    labels: np.ndarray
    original_scores: np.ndarray
    restricted_scores: np.ndarray
    n_patients: int
    reference_max_abs_error: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the matched all-86 INSPECT restriction sensitivity analysis "
            "with shared patient-cluster bootstrap resamples."
        )
    )
    parser.add_argument(
        "--original-npz",
        type=Path,
        default=TRUSTED_PROBE_DIR / "test__f2llm_original_topk__probs.npz",
    )
    parser.add_argument(
        "--restricted-npz",
        type=Path,
        default=TRUSTED_PROBE_DIR / "test__f2llm_trusted__probs.npz",
    )
    parser.add_argument(
        "--manifest-parquet",
        type=Path,
        default=ROOT / "phenotype_labels/per_ct/manifest.parquet",
    )
    parser.add_argument(
        "--labels-parquet",
        type=Path,
        default=ROOT / "phenotype_labels/per_ct/per_ct_labels_visit_only.parquet",
    )
    parser.add_argument(
        "--concept-json",
        type=Path,
        default=(
            ROOT
            / "experiments/exp4_confounder_audit/results/"
            "f2llm_refinement_all86_concepts.json"
        ),
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=TRUSTED_PROBE_DIR / "trusted_vs_baselines.csv",
        help="Existing point-estimate table used only as a provenance cross-check.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument(
        "--plate-concepts-per-profile",
        type=int,
        default=1,
        help="Maximum stored positive concepts shown per profile in each example panel.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Required input files are missing:\n  " + "\n  ".join(missing))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def load_prediction_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as payload:
        expected = {"probs", "labels", "phecodes", "volumes"}
        missing = expected - set(payload.files)
        if missing:
            raise ValueError(f"{path} is missing NPZ keys: {sorted(missing)}")
        return {
            "probs": np.asarray(payload["probs"], dtype=np.float64),
            "labels": [str(value) for value in payload["labels"]],
            "phecodes": [str(value) for value in payload["phecodes"]],
            "volumes": np.asarray([str(value) for value in payload["volumes"]]),
        }


def load_data(args: argparse.Namespace) -> LoadedData:
    require_files(
        [
            args.original_npz,
            args.restricted_npz,
            args.manifest_parquet,
            args.labels_parquet,
            args.concept_json,
            args.reference_summary,
        ]
    )

    original = load_prediction_npz(args.original_npz)
    restricted = load_prediction_npz(args.restricted_npz)

    for key in ("labels", "phecodes"):
        if original[key] != restricted[key]:
            raise ValueError(f"Prediction NPZ {key} are not identically aligned")
    if not np.array_equal(original["volumes"], restricted["volumes"]):
        raise ValueError("Prediction NPZ volumes are not identically aligned")
    if original["probs"].shape != restricted["probs"].shape:
        raise ValueError("Prediction matrices have different shapes")
    if original["probs"].shape != (
        len(original["volumes"]),
        len(original["phecodes"]),
    ):
        raise ValueError("Prediction matrix shape does not match volume/phecode axes")
    phenotype_records = json.loads(args.concept_json.read_text())
    if not isinstance(phenotype_records, list) or len(phenotype_records) != 86:
        raise ValueError("Concept JSON must contain exactly 86 phenotype records")
    phecodes = [str(record["phecode"]) for record in phenotype_records]
    phenotype_names = [str(record["phenotype"]) for record in phenotype_records]
    if len(set(phecodes)) != 86 or len(set(phenotype_names)) != 86:
        raise ValueError("Concept JSON phenotype names and phecodes must be unique")

    npz_index = {phecode: index for index, phecode in enumerate(original["phecodes"])}
    missing_phecodes = sorted(set(phecodes) - set(npz_index))
    if missing_phecodes:
        raise ValueError(f"Concept JSON phecodes missing from prediction NPZ: {missing_phecodes}")
    selected_columns = [npz_index[phecode] for phecode in phecodes]
    for name, phecode, column in zip(phenotype_names, phecodes, selected_columns):
        if original["labels"][column] != name:
            raise ValueError(
                f"Phenotype label mismatch for {phecode}: concept JSON={name!r}, "
                f"prediction NPZ={original['labels'][column]!r}"
            )

    volumes = original["volumes"]
    if len(np.unique(volumes)) != len(volumes):
        raise ValueError("Prediction NPZ volume names are not unique")

    manifest_columns = ["person_id", "VolumeName", "split", "visit_occurrence_id"]
    manifest = pd.read_parquet(args.manifest_parquet, columns=manifest_columns)
    matched_manifest = manifest[manifest["VolumeName"].isin(volumes)].copy()
    if matched_manifest["VolumeName"].duplicated().any():
        duplicated = matched_manifest.loc[
            matched_manifest["VolumeName"].duplicated(), "VolumeName"
        ].tolist()
        raise ValueError(f"Manifest has duplicate rows for prediction volumes: {duplicated[:5]}")
    if len(matched_manifest) != len(volumes):
        found = set(matched_manifest["VolumeName"])
        missing = [volume for volume in volumes if volume not in found]
        raise ValueError(f"Prediction volumes missing from INSPECT manifest: {missing[:5]}")
    if set(matched_manifest["split"].astype(str)) != {"test"}:
        raise ValueError("Prediction volumes are not all assigned to the INSPECT test split")
    if matched_manifest["visit_occurrence_id"].isna().any():
        raise ValueError("At least one prediction volume lacks a visit_occurrence_id")

    manifest_by_volume = matched_manifest.set_index("VolumeName")
    person_ids = manifest_by_volume.loc[volumes, "person_id"].astype(str).to_numpy()
    if np.any(person_ids == "") or np.any(person_ids == "nan"):
        raise ValueError("At least one prediction volume lacks a usable person_id")

    labels_frame = pd.read_parquet(args.labels_parquet, columns=phecodes)
    if not labels_frame.index.is_unique:
        raise ValueError("INSPECT label table VolumeName index is not unique")
    missing_label_volumes = [volume for volume in volumes if volume not in labels_frame.index]
    if missing_label_volumes:
        raise ValueError(
            f"Prediction volumes missing from INSPECT label table: {missing_label_volumes[:5]}"
        )
    labels = labels_frame.loc[volumes, phecodes].to_numpy(dtype=np.uint8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("INSPECT phenotype labels are not binary")
    positive_counts = labels.sum(axis=0)
    if np.any(positive_counts == 0) or np.any(positive_counts == len(labels)):
        raise ValueError("At least one selected phenotype has only one class in the test set")

    original_scores = original["probs"][:, selected_columns]
    restricted_scores = restricted["probs"][:, selected_columns]
    for profile, scores in (
        (ORIGINAL_PROFILE, original_scores),
        (RESTRICTED_PROFILE, restricted_scores),
    ):
        if not np.isfinite(scores).all():
            raise ValueError(
                f"{profile} predictions contain non-finite values within the selected "
                "all-86 phenotype subset"
            )

    # Cross-check the fixed prediction files against the retained point-estimate table.
    reference = pd.read_csv(args.reference_summary)
    reference_by_label = reference.set_index("label")
    reference_errors: dict[str, list[float]] = {
        ORIGINAL_PROFILE: [],
        RESTRICTED_PROFILE: [],
    }
    for index, name in enumerate(phenotype_names):
        if name not in reference_by_label.index:
            raise ValueError(f"Reference summary is missing phenotype {name!r}")
        y = labels[:, index]
        point_original = float(roc_auc_score(y, original_scores[:, index]))
        point_restricted = float(roc_auc_score(y, restricted_scores[:, index]))
        row = reference_by_label.loc[name]
        reference_errors[ORIGINAL_PROFILE].append(
            abs(point_original - float(row["auc_f2llm_original_topk"]))
        )
        reference_errors[RESTRICTED_PROFILE].append(
            abs(point_restricted - float(row["auc_f2llm_trusted"]))
        )
    max_reference_error = {
        profile: float(max(errors)) for profile, errors in reference_errors.items()
    }
    if max(max_reference_error.values()) > 1e-12:
        raise ValueError(
            "Prediction-derived point AUROCs do not reproduce trusted_vs_baselines.csv: "
            f"{max_reference_error}"
        )

    display_phenotype_names = [
        DISPLAY_PHENOTYPE_CORRECTIONS.get(name, name) for name in phenotype_names
    ]

    return LoadedData(
        phenotype_records=phenotype_records,
        phenotype_names=display_phenotype_names,
        phecodes=phecodes,
        volumes=volumes,
        person_ids=person_ids,
        labels=labels,
        original_scores=original_scores,
        restricted_scores=restricted_scores,
        n_patients=int(len(np.unique(person_ids))),
        reference_max_abs_error=max_reference_error,
    )


def build_shared_cluster_weights(
    person_ids: np.ndarray, n_bootstrap: int, seed: int
) -> tuple[np.ndarray, int]:
    """Return bootstrap multiplicity per volume for shared patient resamples."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    unique_people, volume_to_patient = np.unique(person_ids, return_inverse=True)
    n_patients = len(unique_people)
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_patients, 1.0 / n_patients, dtype=np.float64)
    patient_multiplicity = rng.multinomial(
        n_patients, probabilities, size=n_bootstrap
    )
    if patient_multiplicity.max() > np.iinfo(np.uint16).max:
        raise ValueError("Unexpectedly large bootstrap cluster multiplicity")
    volume_weights = patient_multiplicity[:, volume_to_patient].astype(np.uint16)
    if not np.all(volume_weights.sum(axis=1) >= n_patients):
        raise AssertionError("Each patient bootstrap should retain at least n_patients volumes")
    return volume_weights, n_patients


def weighted_auc_for_all_resamples(
    y_true: np.ndarray, scores: np.ndarray, volume_weights: np.ndarray
) -> np.ndarray:
    """Vectorized weighted AUROC for all bootstrap replicates, with exact tie handling."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = np.asarray(scores[order])
    sorted_y = np.asarray(y_true[order], dtype=np.float64)
    weights = np.asarray(volume_weights[:, order], dtype=np.float64)

    positive = weights * sorted_y[None, :]
    negative = weights - positive
    positive_cumulative = np.cumsum(positive, axis=1)
    negative_cumulative = np.cumsum(negative, axis=1)

    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(sorted_scores) - 1]
    n_resamples = len(volume_weights)
    before_positive = np.zeros((n_resamples, len(starts)), dtype=np.float64)
    before_negative = np.zeros((n_resamples, len(starts)), dtype=np.float64)
    nonzero_starts = starts > 0
    before_positive[:, nonzero_starts] = positive_cumulative[
        :, starts[nonzero_starts] - 1
    ]
    before_negative[:, nonzero_starts] = negative_cumulative[
        :, starts[nonzero_starts] - 1
    ]
    group_positive = positive_cumulative[:, ends] - before_positive
    group_negative = negative_cumulative[:, ends] - before_negative

    concordant_weight = np.sum(
        group_positive * (before_negative + 0.5 * group_negative), axis=1
    )
    total_positive = positive_cumulative[:, -1]
    total_negative = negative_cumulative[:, -1]
    denominator = total_positive * total_negative
    auc = np.full(n_resamples, np.nan, dtype=np.float64)
    np.divide(concordant_weight, denominator, out=auc, where=denominator > 0)
    return auc


def validate_weighted_auc_implementation(
    data: LoadedData,
    volume_weights: np.ndarray,
    original_bootstrap: np.ndarray,
    restricted_bootstrap: np.ndarray,
) -> float:
    """Compare a small deterministic subset against sklearn's sample-weighted AUROC."""
    label_indices = sorted({0, len(data.phecodes) // 2, len(data.phecodes) - 1})
    bootstrap_indices = sorted({0, min(1, len(volume_weights) - 1)})
    maximum_error = 0.0
    for label_index in label_indices:
        y = data.labels[:, label_index]
        for bootstrap_index in bootstrap_indices:
            weights = volume_weights[bootstrap_index].astype(np.float64)
            if y[weights > 0].min() == y[weights > 0].max():
                continue
            expected_original = roc_auc_score(
                y, data.original_scores[:, label_index], sample_weight=weights
            )
            expected_restricted = roc_auc_score(
                y, data.restricted_scores[:, label_index], sample_weight=weights
            )
            maximum_error = max(
                maximum_error,
                abs(expected_original - original_bootstrap[bootstrap_index, label_index]),
                abs(expected_restricted - restricted_bootstrap[bootstrap_index, label_index]),
            )
    if maximum_error > 1e-12:
        raise AssertionError(
            f"Vectorized weighted AUROC validation failed; max error={maximum_error:.3g}"
        )
    return float(maximum_error)


def percentile_interval(values: np.ndarray, ci_level: float) -> tuple[float, float, int]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan"), 0
    alpha = 1.0 - ci_level
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high), int(len(finite))


def run_analysis(
    data: LoadedData,
    volume_weights: np.ndarray,
    ci_level: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, float]:
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be strictly between zero and one")
    n_bootstrap = len(volume_weights)
    n_labels = len(data.phecodes)
    original_bootstrap = np.empty((n_bootstrap, n_labels), dtype=np.float64)
    restricted_bootstrap = np.empty((n_bootstrap, n_labels), dtype=np.float64)
    point_original = np.empty(n_labels, dtype=np.float64)
    point_restricted = np.empty(n_labels, dtype=np.float64)

    for index, (name, phecode) in enumerate(zip(data.phenotype_names, data.phecodes)):
        y = data.labels[:, index]
        point_original[index] = roc_auc_score(y, data.original_scores[:, index])
        point_restricted[index] = roc_auc_score(y, data.restricted_scores[:, index])
        original_bootstrap[:, index] = weighted_auc_for_all_resamples(
            y, data.original_scores[:, index], volume_weights
        )
        restricted_bootstrap[:, index] = weighted_auc_for_all_resamples(
            y, data.restricted_scores[:, index], volume_weights
        )
        if (index + 1) % 10 == 0 or index + 1 == n_labels:
            print(f"  bootstrap AUROCs: {index + 1:>2}/{n_labels} phenotypes", flush=True)

    delta_bootstrap = restricted_bootstrap - original_bootstrap
    implementation_error = validate_weighted_auc_implementation(
        data, volume_weights, original_bootstrap, restricted_bootstrap
    )

    rows: list[dict[str, Any]] = []
    for index, (name, phecode) in enumerate(zip(data.phenotype_names, data.phecodes)):
        original_low, original_high, original_valid = percentile_interval(
            original_bootstrap[:, index], ci_level
        )
        restricted_low, restricted_high, restricted_valid = percentile_interval(
            restricted_bootstrap[:, index], ci_level
        )
        delta_low, delta_high, delta_valid = percentile_interval(
            delta_bootstrap[:, index], ci_level
        )
        n_positive = int(data.labels[:, index].sum())
        rows.append(
            {
                "phenotype": name,
                "phecode": phecode,
                "n_test_patients": data.n_patients,
                "n_test_volumes": len(data.volumes),
                "n_positive_volumes": n_positive,
                "n_negative_volumes": int(len(data.volumes) - n_positive),
                "original_point_auroc": float(point_original[index]),
                "original_ci_low": original_low,
                "original_ci_high": original_high,
                "original_n_bootstrap_valid": original_valid,
                "restricted_point_auroc": float(point_restricted[index]),
                "restricted_ci_low": restricted_low,
                "restricted_ci_high": restricted_high,
                "restricted_n_bootstrap_valid": restricted_valid,
                "delta_auroc": float(point_restricted[index] - point_original[index]),
                "delta_ci_low": delta_low,
                "delta_ci_high": delta_high,
                "delta_n_bootstrap_valid": delta_valid,
            }
        )

    summary = pd.DataFrame(rows)
    ordered = summary.sort_values(
        ["delta_auroc", "phenotype", "phecode"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    rank_by_index = {index: rank for rank, index in enumerate(ordered.index, start=1)}
    summary["rank_by_delta"] = [rank_by_index[index] for index in summary.index]
    summary = summary.sort_values("rank_by_delta").reset_index(drop=True)
    return (
        summary,
        original_bootstrap,
        restricted_bootstrap,
        delta_bootstrap,
        implementation_error,
    )


def select_balanced_examples(summary: pd.DataFrame) -> pd.DataFrame:
    gains = summary[summary["delta_auroc"] > 0].sort_values(
        ["delta_auroc", "phenotype", "phecode"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    declines = summary[summary["delta_auroc"] < 0].sort_values(
        ["delta_auroc", "phenotype", "phecode"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    if len(gains) < 4 or len(declines) < 4:
        raise ValueError("Need at least four positive and four negative point deltas")
    top = gains.head(4).copy()
    top["example_group"] = "top positive point delta"
    top["example_rank_within_group"] = np.arange(1, 5)
    bottom = declines.head(4).copy()
    bottom["example_group"] = "most negative point delta"
    bottom["example_rank_within_group"] = np.arange(1, 5)
    return pd.concat([top, bottom], ignore_index=True)


def make_tidy_table(
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    ci_level: float,
    n_bootstrap: int,
) -> pd.DataFrame:
    selected_lookup = selected.set_index("phecode")[[
        "example_group",
        "example_rank_within_group",
    ]].to_dict("index")
    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        selection = selected_lookup.get(row.phecode, {})
        common = {
            "phenotype": row.phenotype,
            "phecode": row.phecode,
            "ci_level": ci_level,
            "bootstrap_method": BOOTSTRAP_METHOD,
            "n_bootstrap_requested": n_bootstrap,
            "n_test_patients": row.n_test_patients,
            "n_test_volumes": row.n_test_volumes,
            "n_positive_volumes": row.n_positive_volumes,
            "n_negative_volumes": row.n_negative_volumes,
            "rank_by_delta": row.rank_by_delta,
            "example_group": selection.get("example_group", ""),
            "example_rank_within_group": selection.get(
                "example_rank_within_group", ""
            ),
        }
        rows.extend(
            [
                {
                    **common,
                    "estimand": "AUROC",
                    "comparator": ORIGINAL_PROFILE,
                    "reference_profile": "",
                    "estimate": row.original_point_auroc,
                    "ci_low": row.original_ci_low,
                    "ci_high": row.original_ci_high,
                    "n_bootstrap_valid": row.original_n_bootstrap_valid,
                },
                {
                    **common,
                    "estimand": "AUROC",
                    "comparator": RESTRICTED_PROFILE,
                    "reference_profile": "",
                    "estimate": row.restricted_point_auroc,
                    "ci_low": row.restricted_ci_low,
                    "ci_high": row.restricted_ci_high,
                    "n_bootstrap_valid": row.restricted_n_bootstrap_valid,
                },
                {
                    **common,
                    "estimand": "delta AUROC",
                    "comparator": RESTRICTED_PROFILE,
                    "reference_profile": ORIGINAL_PROFILE,
                    "estimate": row.delta_auroc,
                    "ci_low": row.delta_ci_low,
                    "ci_high": row.delta_ci_high,
                    "n_bootstrap_valid": row.delta_n_bootstrap_valid,
                },
            ]
        )
    columns = [
        "phenotype",
        "phecode",
        "estimand",
        "comparator",
        "reference_profile",
        "estimate",
        "ci_level",
        "ci_low",
        "ci_high",
        "bootstrap_method",
        "n_bootstrap_requested",
        "n_bootstrap_valid",
        "n_test_patients",
        "n_test_volumes",
        "n_positive_volumes",
        "n_negative_volumes",
        "rank_by_delta",
        "example_group",
        "example_rank_within_group",
    ]
    return pd.DataFrame(rows)[columns]


LATEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value: Any) -> str:
    text = str(value)
    return "".join(LATEX_ESCAPE.get(character, character) for character in text)


def format_ci(point: float, low: float, high: float, signed: bool = False) -> str:
    point_text = f"{point:+.3f}" if signed else f"{point:.3f}"
    # Use an intentional two-line cell in the portrait table so the estimates
    # remain readable at the manuscript's 12 pt body size without CI wrapping.
    return rf"\shortstack[c]{{{point_text}\\({low:.3f}, {high:.3f})}}"


def write_latex_longtable(
    summary: pd.DataFrame, path: Path, ci_level: float, n_bootstrap: int
) -> None:
    ci_percent = int(round(ci_level * 100))
    lines = [
        "% Auto-generated by matched_restriction_sensitivity.py.",
        "% Portrait-only; requires \\usepackage{array,booktabs,longtable}.",
        r"\begingroup",
        r"\normalsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt plus 1fill}",
        (
            r"\begin{longtable}{@{}"
            r">{\raggedright\arraybackslash}p{0.305\linewidth}"
            r">{\centering\arraybackslash}p{0.090\linewidth}"
            r">{\centering\arraybackslash}p{0.195\linewidth}"
            r">{\centering\arraybackslash}p{0.195\linewidth}"
            r">{\centering\arraybackslash}p{0.180\linewidth}@{}}"
        ),
        (
            r"\caption{\textbf{Matched restriction sensitivity across 86 INSPECT phenotypes.} "
            rf"Intervals are {ci_percent}\% percentile intervals from {n_bootstrap:,} "
            r"shared patient-cluster bootstrap resamples. Rows are ordered by point "
            r"change in AUROC.}\label{tab:matched-restriction-sensitivity}\\"
        ),
        r"\toprule",
        (
            r"\textbf{Phenotype} & \textbf{Phecode} & "
            rf"\textbf{{\shortstack[c]{{original top-100\\profile\\AUROC ({ci_percent}\% CI)}}}} & "
            rf"\textbf{{\shortstack[c]{{rule-restricted\\profile\\AUROC ({ci_percent}\% CI)}}}} & "
            rf"\textbf{{\shortstack[c]{{$\Delta$ AUROC\\({ci_percent}\% CI)}}}} \\"
        ),
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{l}{\tablename\ \thetable\ -- continued from previous page}\\",
        r"\toprule",
        (
            r"\textbf{Phenotype} & \textbf{Phecode} & "
            rf"\textbf{{\shortstack[c]{{original top-100\\profile\\AUROC ({ci_percent}\% CI)}}}} & "
            rf"\textbf{{\shortstack[c]{{rule-restricted\\profile\\AUROC ({ci_percent}\% CI)}}}} & "
            rf"\textbf{{\shortstack[c]{{$\Delta$ AUROC\\({ci_percent}\% CI)}}}} \\"
        ),
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{5}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    latex_escape(row.phenotype),
                    latex_escape(row.phecode),
                    format_ci(
                        row.original_point_auroc,
                        row.original_ci_low,
                        row.original_ci_high,
                    ),
                    format_ci(
                        row.restricted_point_auroc,
                        row.restricted_ci_low,
                        row.restricted_ci_high,
                    ),
                    format_ci(
                        row.delta_auroc,
                        row.delta_ci_low,
                        row.delta_ci_high,
                        signed=True,
                    ),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    path.write_text("\n".join(lines))


def save_figure(fig: plt.Figure, png: Path, pdf: Path, dpi: int) -> None:
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "matched_restriction_sensitivity.py", "CreationDate": None},
    )
    plt.close(fig)


WATERFALL_DISPLAY_NAMES = {
    "Nonspecific abnormal findings on radiological and other examination of musculoskeletal system": "Nonspecific musculoskeletal imaging findings",
    "Heart failure with reduced EF [Systolic or combined heart failure]": "Heart failure with reduced EF",
    "Pulmonary collapse; interstitial and compensatory emphysema": "Pulmonary collapse / interstitial emphysema",
    "Heart failure with preserved EF [Diastolic heart failure]": "Heart failure with preserved EF",
    "Pneumonitis due to inhalation of food or vomitus": "Aspiration pneumonitis",
    "Overweight, obesity and other hyperalimentation": "Overweight / obesity",
    "Systemic inflammatory response syndrome (SIRS)": "Systemic inflammatory response syndrome",
    "Cirrhosis of liver without mention of alcohol": "Cirrhosis without alcohol",
    "Secondary malignancy of respiratory organs": "Secondary respiratory-organ malignancy",
    "Type 2 diabetes with renal manifestations": "T2D with renal manifestations",
}


def waterfall_display_label(phenotype: str, phecode: str) -> str:
    """Return a deterministic compact display label; full names remain in the tables."""
    display = WATERFALL_DISPLAY_NAMES.get(phenotype, phenotype)
    display = textwrap.shorten(display, width=48, placeholder="...")
    return f"{display} ({phecode})"


def plot_delta_waterfall(
    summary: pd.DataFrame,
    png: Path,
    pdf: Path,
    ci_level: float,
    n_bootstrap: int,
    dpi: int,
) -> None:
    ordered = summary.sort_values(
        ["delta_auroc", "phenotype", "phecode"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["delta_rank"] = np.arange(1, len(ordered) + 1)
    halves = [ordered.iloc[:43].copy(), ordered.iloc[43:].copy()]
    all_low = float(ordered["delta_ci_low"].min())
    all_high = float(ordered["delta_ci_high"].max())
    padding = 0.035 * (all_high - all_low)
    shared_xlim = (all_low - padding, all_high + padding)

    # Render directly at the intended landscape inclusion size. With 43 rows
    # per half, 7.2-point labels retain approximately 1.2 points of line gap.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.5), sharex=True)
    for half_index, (ax, half) in enumerate(zip(axes, halves)):
        y = np.arange(len(half))
        delta = half["delta_auroc"].to_numpy()
        low = half["delta_ci_low"].to_numpy()
        high = half["delta_ci_high"].to_numpy()
        colors = np.where(
            delta > 0,
            "#2C7FB8",
            np.where(delta < 0, "#D97841", "#777777"),
        )
        ax.barh(y, delta, color=colors, alpha=0.85, edgecolor="none", height=0.72)
        ax.errorbar(
            delta,
            y,
            xerr=np.vstack((delta - low, high - delta)),
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=1.9,
            capthick=0.8,
            zorder=3,
        )
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [
                f"{int(row.delta_rank)}. "
                + waterfall_display_label(row.phenotype, row.phecode)
                for row in half.itertuples()
            ],
            fontsize=7.2,
        )
        ax.invert_yaxis()
        ax.set_xlim(*shared_xlim)
        start_rank = int(half["delta_rank"].min())
        end_rank = int(half["delta_rank"].max())
        ax.set_title(
            f"Ordered ranks {start_rank}–{end_rank}",
            loc="left",
            fontsize=8.5,
            fontweight="bold",
            pad=8,
        )
        ax.xaxis.grid(True, color="0.88", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.tick_params(axis="x", labelsize=7.0)
        if half_index == 1:
            ax.yaxis.set_tick_params(pad=3)

    ci_percent = int(round(ci_level * 100))
    fig.supxlabel(
        f"Change in AUROC: {RESTRICTED_PROFILE} minus {ORIGINAL_PROFILE}\n"
        f"point estimate with {ci_percent}% patient-cluster percentile CI",
        fontsize=8.0,
        y=0.050,
    )
    fig.suptitle(
        "Matched restriction sensitivity across 86 INSPECT phenotypes",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.01,
        0.004,
        (
            f"{n_bootstrap:,} shared patient-cluster bootstrap resamples; "
            f"{int(summary.n_test_patients.iloc[0]):,} patients and "
            f"{int(summary.n_test_volumes.iloc[0]):,} volumes. "
            "Intervals describe test-cohort sampling variability and do not include retraining."
        ),
        ha="left",
        va="bottom",
        fontsize=7.0,
    )
    fig.subplots_adjust(
        left=0.265,
        right=0.990,
        top=0.925,
        bottom=0.145,
        wspace=1.20,
    )
    save_figure(fig, png, pdf, dpi)


def concept_block(concepts: list[str], maximum: int) -> str:
    if not concepts:
        return "No stored positive concept"
    blocks: list[str] = []
    for index, concept in enumerate(concepts[:maximum], start=1):
        shortened = textwrap.shorten(str(concept), width=42, placeholder="...")
        wrapped = textwrap.wrap(shortened, width=19, break_long_words=False)[:2]
        if not wrapped:
            continue
        blocks.append(f"{index}. " + "\n   ".join(wrapped))
    return "\n".join(blocks)


def plot_balanced_example_plate(
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    records: list[dict[str, Any]],
    png: Path,
    pdf: Path,
    ci_level: float,
    maximum_concepts: int,
    dpi: int,
) -> None:
    record_by_phecode = {str(record["phecode"]): record for record in records}
    selected = selected.copy()
    selected["row_order"] = selected["example_group"].map(
        {"top positive point delta": 0, "most negative point delta": 1}
    )
    selected = selected.sort_values(
        ["row_order", "example_rank_within_group"], kind="mergesort"
    )

    # Render at the intended landscape inclusion size; no downscaling is
    # required to fit a 10.5 x 6.5 inch supplementary page region.
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 6.5))
    ci_percent = int(round(ci_level * 100))
    original_color = "#D97841"
    restricted_color = "#2C7FB8"

    for panel_index, (axis, row) in enumerate(
        zip(axes.flat, selected.itertuples(index=False))
    ):
        axis.set_axis_off()
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.text(
            0.0,
            0.99,
            chr(ord("a") + panel_index),
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        axis.text(
            0.09,
            0.99,
            textwrap.fill(row.phenotype, width=24),
            ha="left",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            linespacing=0.92,
        )
        axis.text(
            0.09,
            0.835,
            (
                f"Phecode {row.phecode}; $\\Delta$AUROC {row.delta_auroc:+.3f} "
                f"[{row.delta_ci_low:+.3f}, {row.delta_ci_high:+.3f}]"
            ),
            ha="left",
            va="top",
            fontsize=7.0,
        )

        auc_axis = axis.inset_axes([0.08, 0.695, 0.86, 0.105])
        points = np.array([row.original_point_auroc, row.restricted_point_auroc])
        lows = np.array([row.original_ci_low, row.restricted_ci_low])
        highs = np.array([row.original_ci_high, row.restricted_ci_high])
        auc_axis.errorbar(
            points,
            [0, 0],
            xerr=np.vstack((points - lows, highs - points)),
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
            zorder=2,
        )
        auc_axis.scatter(
            points,
            [0, 0],
            color=[original_color, restricted_color],
            s=22,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        auc_axis.plot(points, [0, 0], color="0.55", linewidth=0.7, zorder=1)
        auc_axis.set_xlim(0.50, 1.00)
        auc_axis.set_ylim(-0.5, 0.5)
        auc_axis.set_yticks([])
        auc_axis.set_xticks([0.5, 0.75, 1.0])
        auc_axis.set_xlabel("AUROC", fontsize=6.5, labelpad=0.5)
        auc_axis.tick_params(axis="x", labelsize=6.3, pad=1.0)
        auc_axis.xaxis.grid(True, color="0.90", linewidth=0.5)
        for spine in ("top", "right", "left"):
            auc_axis.spines[spine].set_visible(False)

        record = record_by_phecode[row.phecode]
        axis.text(
            0.02,
            0.55,
            "Original\nconcept",
            color=original_color,
            fontsize=7.0,
            fontweight="bold",
            ha="left",
            va="top",
            linespacing=0.90,
        )
        axis.text(
            0.53,
            0.55,
            "Restricted\nconcept",
            color=restricted_color,
            fontsize=7.0,
            fontweight="bold",
            ha="left",
            va="top",
            linespacing=0.90,
        )
        axis.text(
            0.02,
            0.415,
            concept_block(record.get("prior_top6_pos_concepts", []), maximum_concepts),
            fontsize=7.0,
            ha="left",
            va="top",
            linespacing=0.92,
            color="0.12",
        )
        axis.text(
            0.53,
            0.415,
            concept_block(record.get("after_top6_pos_concepts", []), maximum_concepts),
            fontsize=7.0,
            ha="left",
            va="top",
            linespacing=0.92,
            color="0.12",
        )
        axis.plot([0.50, 0.50], [0.08, 0.57], color="0.88", linewidth=0.7)

    fig.text(
        0.005,
        0.940,
        "Top four positive point deltas",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color="#2C7FB8",
    )
    fig.text(
        0.005,
        0.485,
        "Four most negative point deltas",
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color="#D97841",
    )
    fig.suptitle(
        "Deterministic balanced examples from the matched restriction sensitivity analysis",
        x=0.5,
        y=0.995,
        fontsize=11,
        fontweight="bold",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=original_color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=5,
            label=ORIGINAL_PROFILE,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=restricted_color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=5,
            label=RESTRICTED_PROFILE,
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.73, 0.962),
        ncol=2,
        frameon=False,
        fontsize=7.2,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.text(
        0.5,
        0.006,
        (
            f"Selection uses unrounded point change in AUROC only: four largest positive "
            f"and four most negative values, with phenotype and phecode as deterministic "
            f"tie-breakers. Error bars are {ci_percent}% patient-cluster percentile CIs.\n"
            "Stored positive concepts are descriptive profile examples, not clinical validation."
        ),
        ha="center",
        va="bottom",
        fontsize=7.0,
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.995,
        top=0.900,
        bottom=0.095,
        hspace=0.32,
        wspace=0.16,
    )
    save_figure(fig, png, pdf, dpi)


def write_selected_examples(
    selected: pd.DataFrame,
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    record_by_phecode = {str(record["phecode"]): record for record in records}
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        record = record_by_phecode[row.phecode]
        rows.append(
            {
                "example_group": row.example_group,
                "example_rank_within_group": row.example_rank_within_group,
                "phenotype": row.phenotype,
                "phecode": row.phecode,
                "original_point_auroc": row.original_point_auroc,
                "rule_restricted_point_auroc": row.restricted_point_auroc,
                "delta_auroc": row.delta_auroc,
                "delta_ci_low": row.delta_ci_low,
                "delta_ci_high": row.delta_ci_high,
                "original_top100_positive_concepts_json": json.dumps(
                    record.get("prior_top6_pos_concepts", []), ensure_ascii=False
                ),
                "rule_restricted_positive_concepts_json": json.dumps(
                    record.get("after_top6_pos_concepts", []), ensure_ascii=False
                ),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def git_state() -> dict[str, Any]:
    def run(*command: str) -> str:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "branch", "--show-current"),
            "working_tree_dirty": bool(run("git", "status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "working_tree_dirty": None}


def write_status(
    args: argparse.Namespace,
    data: LoadedData,
    summary: pd.DataFrame,
    original_bootstrap: np.ndarray,
    restricted_bootstrap: np.ndarray,
    delta_bootstrap: np.ndarray,
    selected: pd.DataFrame,
    input_paths: list[Path],
    output_paths: list[Path],
    implementation_error: float,
    elapsed_seconds: float,
    path: Path,
) -> None:
    # Macro-average each shared patient-cluster resample across the same 86
    # phenotypes. Taking the per-replicate mean before differencing preserves
    # the pairing across profiles, phenotypes, and patient draws.
    original_mean_bootstrap = np.mean(original_bootstrap, axis=1)
    restricted_mean_bootstrap = np.mean(restricted_bootstrap, axis=1)
    paired_mean_delta_bootstrap = np.mean(delta_bootstrap, axis=1)
    original_mean_low, original_mean_high, original_mean_valid = percentile_interval(
        original_mean_bootstrap, args.ci_level
    )
    restricted_mean_low, restricted_mean_high, restricted_mean_valid = percentile_interval(
        restricted_mean_bootstrap, args.ci_level
    )
    mean_delta_low, mean_delta_high, mean_delta_valid = percentile_interval(
        paired_mean_delta_bootstrap, args.ci_level
    )
    selection_records = selected[
        [
            "example_group",
            "example_rank_within_group",
            "phenotype",
            "phecode",
            "delta_auroc",
        ]
    ].to_dict("records")
    payload = {
        "schema_version": 1,
        "analysis": "matched restriction sensitivity across 86 INSPECT phenotypes",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {
            "path": display_path(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "git": git_state(),
        "comparator_labels": {
            "original": ORIGINAL_PROFILE,
            "restricted": RESTRICTED_PROFILE,
        },
        "parameters": {
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "ci_level": args.ci_level,
            "bootstrap_method": BOOTSTRAP_METHOD,
            "plate_concepts_per_profile": args.plate_concepts_per_profile,
        },
        "cohort": {
            "split": "test",
            "n_patients": data.n_patients,
            "n_volumes": len(data.volumes),
            "n_phenotypes": len(data.phecodes),
            "patients_with_multiple_volumes": int(
                (pd.Series(data.person_ids).value_counts() > 1).sum()
            ),
            "maximum_volumes_per_patient": int(
                pd.Series(data.person_ids).value_counts().max()
            ),
            "person_ids_written_to_outputs": False,
        },
        "alignment_checks": {
            "prediction_npz_axes_identical": True,
            "all_prediction_volumes_mapped_to_test_manifest": True,
            "all_prediction_volumes_mapped_to_person_id": True,
            "all_labels_binary_and_biclass": True,
            "point_auc_max_abs_error_vs_reference_summary": data.reference_max_abs_error,
            "vectorized_bootstrap_auc_max_abs_error_vs_sklearn": implementation_error,
        },
        "aggregate_results": {
            "positive_point_deltas": int((summary["delta_auroc"] > 0).sum()),
            "zero_point_deltas": int((summary["delta_auroc"] == 0).sum()),
            "negative_point_deltas": int((summary["delta_auroc"] < 0).sum()),
            "mean_point_delta": float(summary["delta_auroc"].mean()),
            "median_point_delta": float(summary["delta_auroc"].median()),
            "minimum_point_delta": float(summary["delta_auroc"].min()),
            "maximum_point_delta": float(summary["delta_auroc"].max()),
            "mean_auroc_across_86_phenotypes": {
                "aggregation": (
                    "unweighted macro mean across the same 86 phenotype AUROCs; "
                    "bootstrap intervals first average within each shared patient-cluster "
                    "resample"
                ),
                "original": {
                    "profile_label": ORIGINAL_PROFILE,
                    "point": float(summary["original_point_auroc"].mean()),
                    "percentile_ci_low": original_mean_low,
                    "percentile_ci_high": original_mean_high,
                    "n_bootstrap_valid": original_mean_valid,
                },
                "restricted": {
                    "profile_label": RESTRICTED_PROFILE,
                    "point": float(summary["restricted_point_auroc"].mean()),
                    "percentile_ci_low": restricted_mean_low,
                    "percentile_ci_high": restricted_mean_high,
                    "n_bootstrap_valid": restricted_mean_valid,
                },
            },
            "paired_mean_delta_across_86_phenotypes": {
                "contrast": f"{RESTRICTED_PROFILE} minus {ORIGINAL_PROFILE}",
                "point": float(summary["delta_auroc"].mean()),
                "percentile_ci_low": mean_delta_low,
                "percentile_ci_high": mean_delta_high,
                "n_bootstrap_valid": mean_delta_valid,
                "bootstrap_unit": (
                    "mean of the 86 paired phenotype deltas within each shared "
                    "patient-cluster resample"
                ),
            },
        },
        "example_selection": {
            "rule": (
                "Select the four largest positive and four most negative unrounded "
                "point changes in AUROC; break ties by phenotype then phecode."
            ),
            "balanced_definition": (
                "Balanced means four positive-delta and four negative-delta examples; "
                "it does not imply clinical or demographic balancing."
            ),
            "selected": selection_records,
        },
        "inputs": [
            {
                "path": display_path(input_path),
                "size_bytes": input_path.stat().st_size,
                "sha256": sha256_file(input_path),
            }
            for input_path in input_paths
        ],
        "outputs": [
            {
                "path": display_path(output_path),
                "size_bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
            for output_path in output_paths
        ],
        "limitations": [
            "This is a fixed-prediction sensitivity analysis; neither profile is retrained in the bootstrap.",
            "Percentile intervals capture test-cohort patient-cluster sampling variability, not training variability.",
            "The same INSPECT test cohort is used for both profiles; this is not external validation.",
            "The rule-restricted profile is not an independently clinically validated concept set.",
            "Stored positive concepts are descriptive profile examples and do not establish causal feature use.",
            "The eight example panels are selected by extreme observed point deltas and must be interpreted with the full 86-phenotype table and waterfall.",
            "No multiplicity-adjusted hypothesis tests are reported.",
        ],
        "elapsed_seconds": elapsed_seconds,
        "command": [sys.executable, *sys.argv],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.plate_concepts_per_profile < 1:
        raise SystemExit("--plate-concepts-per-profile must be positive")
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading and aligning predictions, labels, manifest, and concepts", flush=True)
    data = load_data(args)
    print(
        f"  aligned {len(data.volumes):,} volumes, {data.n_patients:,} patients, "
        f"and {len(data.phecodes)} phenotypes",
        flush=True,
    )

    print("[2/7] Building shared patient-cluster bootstrap multiplicities", flush=True)
    volume_weights, n_patients = build_shared_cluster_weights(
        data.person_ids, args.n_bootstrap, args.seed
    )
    if n_patients != data.n_patients:
        raise AssertionError("Patient count changed while building bootstrap resamples")

    print("[3/7] Computing point and bootstrap AUROCs", flush=True)
    (
        summary,
        original_bootstrap,
        restricted_bootstrap,
        delta_bootstrap,
        implementation_error,
    ) = run_analysis(data, volume_weights, args.ci_level)
    selected = select_balanced_examples(summary)

    tidy_path = args.output_dir / "matched_restriction_sensitivity_tidy.csv"
    longtable_path = args.output_dir / "matched_restriction_sensitivity_longtable.tex"
    bootstrap_path = args.output_dir / "paired_patient_cluster_bootstrap.npz"
    selection_path = args.output_dir / "balanced_example_selection.csv"
    waterfall_png = args.output_dir / "ordered_delta_waterfall.png"
    waterfall_pdf = args.output_dir / "ordered_delta_waterfall.pdf"
    plate_png = args.output_dir / "balanced_examples_2x4.png"
    plate_pdf = args.output_dir / "balanced_examples_2x4.pdf"
    status_path = args.output_dir / "matched_restriction_status.json"
    legacy_status_path = args.output_dir / "status.json"

    print("[4/7] Writing tidy CSV, LaTeX longtable, and bootstrap distributions", flush=True)
    tidy = make_tidy_table(summary, selected, args.ci_level, args.n_bootstrap)
    tidy.to_csv(tidy_path, index=False, float_format="%.10g")
    write_latex_longtable(summary, longtable_path, args.ci_level, args.n_bootstrap)
    write_selected_examples(selected, data.phenotype_records, selection_path)
    np.savez_compressed(
        bootstrap_path,
        phecodes=np.asarray(data.phecodes, dtype=object),
        phenotypes=np.asarray(data.phenotype_names, dtype=object),
        original_profile_label=np.asarray(ORIGINAL_PROFILE),
        restricted_profile_label=np.asarray(RESTRICTED_PROFILE),
        original_auroc=original_bootstrap.astype(np.float32),
        restricted_auroc=restricted_bootstrap.astype(np.float32),
        delta_auroc=delta_bootstrap.astype(np.float32),
        n_bootstrap=np.asarray(args.n_bootstrap),
        seed=np.asarray(args.seed),
        ci_level=np.asarray(args.ci_level),
        n_test_patients=np.asarray(data.n_patients),
        n_test_volumes=np.asarray(len(data.volumes)),
    )

    print("[5/7] Rendering ordered delta waterfall", flush=True)
    plot_delta_waterfall(
        summary,
        waterfall_png,
        waterfall_pdf,
        args.ci_level,
        args.n_bootstrap,
        args.dpi,
    )

    print("[6/7] Rendering deterministic balanced 2x4 example plate", flush=True)
    plot_balanced_example_plate(
        summary,
        selected,
        data.phenotype_records,
        plate_png,
        plate_pdf,
        args.ci_level,
        args.plate_concepts_per_profile,
        args.dpi,
    )

    output_paths = [
        tidy_path,
        longtable_path,
        bootstrap_path,
        selection_path,
        waterfall_png,
        waterfall_pdf,
        plate_png,
        plate_pdf,
    ]
    input_paths = [
        args.original_npz,
        args.restricted_npz,
        args.manifest_parquet,
        args.labels_parquet,
        args.concept_json,
        args.reference_summary,
    ]
    elapsed = time.perf_counter() - started
    print("[7/7] Writing provenance and status", flush=True)
    write_status(
        args,
        data,
        summary,
        original_bootstrap,
        restricted_bootstrap,
        delta_bootstrap,
        selected,
        input_paths,
        output_paths,
        implementation_error,
        elapsed,
        status_path,
    )
    # Keep the original generic filename as a byte-identical compatibility alias.
    legacy_status_path.write_bytes(status_path.read_bytes())

    print(f"Wrote {len(output_paths) + 2} outputs to {args.output_dir}")
    print(
        "Point delta counts: "
        f"{int((summary.delta_auroc > 0).sum())} positive, "
        f"{int((summary.delta_auroc == 0).sum())} zero, "
        f"{int((summary.delta_auroc < 0).sum())} negative"
    )
    print("Selected examples:")
    for row in selected.itertuples(index=False):
        print(
            f"  {row.example_group:>25}: {row.phenotype} ({row.phecode}), "
            f"delta={row.delta_auroc:+.4f}"
        )


if __name__ == "__main__":
    main()

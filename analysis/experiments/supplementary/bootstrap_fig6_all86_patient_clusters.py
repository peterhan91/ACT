#!/usr/bin/env python3
"""Patient-cluster bootstrap for the two Figure 6 all-86 AUROC columns.

The full-bank estimand is the mean AUROC across the 20 retained Adam probe
fits. For every shared patient resample, AUROC is recomputed for each fit and
then averaged across fits. The rule-restricted estimand is the AUROC of the
retained fitted restricted probe. Both intervals therefore describe sampling
variation in the same held-out patient cohort while keeping fitted models
fixed. No person identifier is written to an output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "experiments/exp4_confounder_audit/results"
EXTERNAL = ROOT / "outputs/v1/external"
CACHE = ROOT / "outputs/v1/cache"
TRUSTED = EXTERNAL / "trusted_concept_probe_f2llm_lbfgs"

DEFAULT_LEDGER = RESULTS / "fig6_matched_86_top20_ledger.json"
DEFAULT_REPRESENTATION = CACHE / "llm_repr.inspect_test.f2llm.npy"
DEFAULT_VOLUME_INDEX = CACHE / "volume_index.inspect_test.csv"
DEFAULT_MANIFEST = ROOT / "phenotype_labels/per_ct/manifest.parquet"
DEFAULT_LABELS = ROOT / "phenotype_labels/per_ct/per_ct_labels_visit_only.parquet"
DEFAULT_RESTRICTED = TRUSTED / "test__f2llm_trusted__probs.npz"
DEFAULT_OUTPUT = HERE / "outputs/fig6_all86_patient_bootstrap"
SEED_TEMPLATE = (
    EXTERNAL / "phenotype__linear_f2llm__seed{seed}__lr3e-2__coefs.pt"
)

N_FITS = 20
EXPECTED_N_PHENOTYPES = 86
EXPECTED_N_SCANS = 2_612
EXPECTED_N_PATIENTS = 2_223
EXPECTED_HIGHER = 58
EXPECTED_LOWER = 28
EXPECTED_FULL_MEAN = 0.7405666115048649
EXPECTED_RESTRICTED_MEAN = 0.7513102067776847
POINT_RECONSTRUCTION_TOLERANCE = 1e-5
AGGREGATE_TOLERANCE = 1e-6
DISPLAY_NAME_OVERRIDES = {
    "505": "Other pulmonary inflammation or edema",
}


@dataclass(frozen=True)
class AnalysisData:
    ledger: dict[str, Any]
    rows: list[dict[str, Any]]
    phecodes: list[str]
    source_phenotypes: list[str]
    display_phenotypes: list[str]
    n_retained_observations: np.ndarray
    labels: np.ndarray
    person_ids: np.ndarray
    full_scores: np.ndarray
    restricted_scores: np.ndarray
    full_point_reference: np.ndarray
    restricted_point_reference: np.ndarray
    full_point_reconstructed: np.ndarray
    restricted_point_reconstructed: np.ndarray
    full_per_fit_reference: np.ndarray
    full_per_fit_reconstructed: np.ndarray
    input_paths: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--full-test-representation",
        type=Path,
        default=DEFAULT_REPRESENTATION,
    )
    parser.add_argument("--volume-index", type=Path, default=DEFAULT_VOLUME_INDEX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--restricted-predictions", type=Path, default=DEFAULT_RESTRICTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20_260_721)
    parser.add_argument("--ci-level", type=float, default=0.95)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Figure 6 bootstrap inputs are missing:\n  "
            + "\n  ".join(missing)
        )


def load_data(args: argparse.Namespace) -> AnalysisData:
    ledger_path = args.ledger.resolve()
    representation_path = args.full_test_representation.resolve()
    volume_index_path = args.volume_index.resolve()
    manifest_path = args.manifest.resolve()
    labels_path = args.labels.resolve()
    restricted_path = args.restricted_predictions.resolve()
    seed_paths = [
        Path(str(SEED_TEMPLATE).format(seed=seed)).resolve()
        for seed in range(1, N_FITS + 1)
    ]
    input_paths = [
        ledger_path,
        representation_path,
        volume_index_path,
        manifest_path,
        labels_path,
        restricted_path,
        *seed_paths,
    ]
    require_paths(input_paths)

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    rows = list(ledger["rows"])
    if len(rows) != EXPECTED_N_PHENOTYPES:
        raise ValueError(f"expected 86 ledger rows, found {len(rows)}")
    phecodes = [str(row["phecode"]) for row in rows]
    source_phenotypes = [str(row["phenotype"]) for row in rows]
    if len(set(phecodes)) != len(phecodes):
        raise ValueError("Figure 6 ledger phecodes are not unique")
    display_phenotypes = [
        DISPLAY_NAME_OVERRIDES.get(phecode, phenotype)
        for phecode, phenotype in zip(phecodes, source_phenotypes)
    ]
    n_retained = np.asarray(
        [int(row["n_refined_concepts"]) for row in rows],
        dtype=np.int64,
    )

    manifest = pd.read_parquet(manifest_path)
    labels_frame = pd.read_parquet(labels_path)
    labels_frame.index = labels_frame.index.map(str)
    labels_frame.columns = labels_frame.columns.map(str)
    volume_index = pd.read_csv(volume_index_path)
    cache_volumes = volume_index["VolumeName"].astype(str).tolist()
    cache_lookup = {volume: index for index, volume in enumerate(cache_volumes)}
    if len(cache_lookup) != len(cache_volumes):
        raise ValueError("test cache volume names are not unique")

    test_manifest = manifest[
        (manifest["split"] == "test")
        & manifest["visit_occurrence_id"].notna()
    ].copy()
    test_manifest["VolumeName"] = test_manifest["VolumeName"].astype(str)
    test_manifest = test_manifest.set_index("VolumeName", drop=False)
    volumes = [
        volume
        for volume in test_manifest["VolumeName"].tolist()
        if volume in cache_lookup and volume in labels_frame.index
    ]
    if len(volumes) != EXPECTED_N_SCANS or len(set(volumes)) != len(volumes):
        raise ValueError(
            f"expected {EXPECTED_N_SCANS} unique held-out scans, found {len(volumes)}"
        )
    person_ids = test_manifest.loc[volumes, "person_id"].astype(str).to_numpy()
    if len(np.unique(person_ids)) != EXPECTED_N_PATIENTS:
        raise ValueError(
            f"expected {EXPECTED_N_PATIENTS} held-out patients, "
            f"found {len(np.unique(person_ids))}"
        )
    labels = labels_frame.loc[volumes, phecodes].to_numpy(dtype=np.uint8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("phenotype labels are not binary")
    if np.any(labels.sum(axis=0) == 0) or np.any(labels.sum(axis=0) == len(labels)):
        raise ValueError("at least one phenotype has only one test-set class")

    restricted = np.load(restricted_path, allow_pickle=True)
    restricted_volumes = [str(value) for value in restricted["volumes"]]
    if restricted_volumes != volumes:
        raise ValueError("restricted prediction volumes do not match test order")
    restricted_phecodes = [str(value) for value in restricted["phecodes"]]
    if len(set(restricted_phecodes)) != len(restricted_phecodes):
        raise ValueError("restricted prediction phecodes are not unique")
    restricted_columns = [restricted_phecodes.index(phecode) for phecode in phecodes]
    restricted_scores = np.asarray(
        restricted["probs"][:, restricted_columns],
        dtype=np.float64,
    )

    representation_all = np.load(representation_path, mmap_mode="r")
    if representation_all.shape != (len(cache_volumes), 5_120):
        raise ValueError(
            "unexpected full-bank test representation shape: "
            f"{representation_all.shape}"
        )
    representation = np.asarray(
        representation_all[[cache_lookup[volume] for volume in volumes]],
        dtype=np.float64,
    )

    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    recorded_per_fit: list[list[float]] = []
    label_order: list[str] | None = None
    phecode_order: list[str] | None = None
    for seed, path in enumerate(seed_paths, start=1):
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        labels_this = [str(value) for value in bundle["labels"]]
        phecodes_this = [str(value) for value in bundle["phecodes"]]
        if label_order is None:
            label_order = labels_this
            phecode_order = phecodes_this
        elif labels_this != label_order or phecodes_this != phecode_order:
            raise ValueError(f"full-bank label order changed in seed {seed}")
        if int(bundle["seed"]) != seed or not math.isclose(
            float(bundle["lr"]), 0.03, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"unexpected full-bank seed metadata: {path}")
        assert label_order is not None and phecode_order is not None
        selected = [phecode_order.index(phecode) for phecode in phecodes]
        weights.append(
            bundle["W"].numpy()[selected].astype(np.float64, copy=False)
        )
        biases.append(
            bundle["b"].numpy()[selected].astype(np.float64, copy=False)
        )
        recorded_per_fit.append(
            [
                float(bundle["test_per_label_auc"][label_order[index]])
                for index in selected
            ]
        )

    stacked_weights = np.concatenate(weights, axis=0)
    stacked_biases = np.concatenate(biases, axis=0)
    full_scores = (
        representation @ stacked_weights.T + stacked_biases
    ).reshape(len(volumes), N_FITS, EXPECTED_N_PHENOTYPES).transpose(1, 0, 2)

    full_per_fit_reconstructed = np.empty(
        (N_FITS, EXPECTED_N_PHENOTYPES),
        dtype=np.float64,
    )
    restricted_point_reconstructed = np.empty(
        EXPECTED_N_PHENOTYPES,
        dtype=np.float64,
    )
    for phenotype_index in range(EXPECTED_N_PHENOTYPES):
        y_true = labels[:, phenotype_index]
        for fit_index in range(N_FITS):
            full_per_fit_reconstructed[fit_index, phenotype_index] = roc_auc_score(
                y_true,
                full_scores[fit_index, :, phenotype_index],
            )
        restricted_point_reconstructed[phenotype_index] = roc_auc_score(
            y_true,
            restricted_scores[:, phenotype_index],
        )

    full_per_fit_reference = np.asarray(recorded_per_fit, dtype=np.float64)
    ledger_per_fit = np.asarray(
        [row["full_auc_per_seed"] for row in rows],
        dtype=np.float64,
    ).T
    if np.max(np.abs(full_per_fit_reference - ledger_per_fit)) > 1e-12:
        raise ValueError("seed bundles do not reproduce ledger per-fit AUROCs")
    full_point_reference = np.asarray(
        [float(row["full_auc_mean"]) for row in rows],
        dtype=np.float64,
    )
    restricted_point_reference = np.asarray(
        [float(row["refined_auc_point"]) for row in rows],
        dtype=np.float64,
    )
    max_full_error = float(
        np.max(np.abs(full_per_fit_reconstructed - full_per_fit_reference))
    )
    max_restricted_error = float(
        np.max(np.abs(restricted_point_reconstructed - restricted_point_reference))
    )
    if max_full_error > POINT_RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"full-bank score reconstruction error is too large: {max_full_error}"
        )
    if max_restricted_error > 1e-12:
        raise ValueError(
            f"restricted predictions do not reproduce ledger: {max_restricted_error}"
        )
    if abs(float(full_point_reference.mean()) - EXPECTED_FULL_MEAN) > 1e-12:
        raise ValueError("full-bank aggregate mean changed")
    if (
        abs(float(restricted_point_reference.mean()) - EXPECTED_RESTRICTED_MEAN)
        > 1e-12
    ):
        raise ValueError("rule-restricted aggregate mean changed")

    return AnalysisData(
        ledger=ledger,
        rows=rows,
        phecodes=phecodes,
        source_phenotypes=source_phenotypes,
        display_phenotypes=display_phenotypes,
        n_retained_observations=n_retained,
        labels=labels,
        person_ids=person_ids,
        full_scores=full_scores,
        restricted_scores=restricted_scores,
        full_point_reference=full_point_reference,
        restricted_point_reference=restricted_point_reference,
        full_point_reconstructed=full_per_fit_reconstructed.mean(axis=0),
        restricted_point_reconstructed=restricted_point_reconstructed,
        full_per_fit_reference=full_per_fit_reference,
        full_per_fit_reconstructed=full_per_fit_reconstructed,
        input_paths=input_paths,
    )


def build_shared_cluster_weights(
    person_ids: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    if n_bootstrap < 1:
        raise ValueError("n-bootstrap must be positive")
    unique_people, volume_to_patient = np.unique(person_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    patient_multiplicity = rng.multinomial(
        len(unique_people),
        np.full(len(unique_people), 1.0 / len(unique_people)),
        size=n_bootstrap,
    )
    if patient_multiplicity.max() > np.iinfo(np.uint16).max:
        raise ValueError("unexpectedly large bootstrap multiplicity")
    return patient_multiplicity[:, volume_to_patient].astype(np.uint16)


def weighted_auc_for_all_resamples(
    y_true: np.ndarray,
    scores: np.ndarray,
    volume_weights: np.ndarray,
) -> np.ndarray:
    """Vectorized weighted AUROC with exact handling of tied scores."""
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
    before_positive = np.zeros((len(volume_weights), len(starts)), dtype=np.float64)
    before_negative = np.zeros((len(volume_weights), len(starts)), dtype=np.float64)
    nonzero = starts > 0
    before_positive[:, nonzero] = positive_cumulative[:, starts[nonzero] - 1]
    before_negative[:, nonzero] = negative_cumulative[:, starts[nonzero] - 1]
    group_positive = positive_cumulative[:, ends] - before_positive
    group_negative = negative_cumulative[:, ends] - before_negative
    concordant = np.sum(
        group_positive * (before_negative + 0.5 * group_negative),
        axis=1,
    )
    denominator = positive_cumulative[:, -1] * negative_cumulative[:, -1]
    auc = np.full(len(volume_weights), np.nan, dtype=np.float64)
    np.divide(concordant, denominator, out=auc, where=denominator > 0)
    return auc


def percentile_interval(
    values: np.ndarray,
    ci_level: float,
) -> tuple[float, float, int]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan"), 0
    alpha = 1.0 - ci_level
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high), int(len(finite))


def run_bootstrap(
    data: AnalysisData,
    volume_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_bootstrap = len(volume_weights)
    full_bootstrap = np.empty(
        (n_bootstrap, EXPECTED_N_PHENOTYPES),
        dtype=np.float64,
    )
    restricted_bootstrap = np.empty_like(full_bootstrap)
    for phenotype_index, phenotype in enumerate(data.display_phenotypes):
        y_true = data.labels[:, phenotype_index]
        full_sum = np.zeros(n_bootstrap, dtype=np.float64)
        for fit_index in range(N_FITS):
            full_sum += weighted_auc_for_all_resamples(
                y_true,
                data.full_scores[fit_index, :, phenotype_index],
                volume_weights,
            )
        full_bootstrap[:, phenotype_index] = full_sum / N_FITS
        restricted_bootstrap[:, phenotype_index] = weighted_auc_for_all_resamples(
            y_true,
            data.restricted_scores[:, phenotype_index],
            volume_weights,
        )
        if (phenotype_index + 1) % 10 == 0 or (
            phenotype_index + 1 == EXPECTED_N_PHENOTYPES
        ):
            print(
                f"  bootstrapped {phenotype_index + 1:>2}/"
                f"{EXPECTED_N_PHENOTYPES} phenotypes",
                flush=True,
            )

    if not np.isfinite(full_bootstrap).all():
        raise ValueError("a full-bank bootstrap replicate lacked one outcome class")
    if not np.isfinite(restricted_bootstrap).all():
        raise ValueError("a restricted bootstrap replicate lacked one outcome class")

    validation_indices = [
        (0, 0),
        (len(volume_weights) - 1, EXPECTED_N_PHENOTYPES - 1),
    ]
    maximum_error = 0.0
    for bootstrap_index, phenotype_index in validation_indices:
        weights = volume_weights[bootstrap_index].astype(np.float64)
        y_true = data.labels[:, phenotype_index]
        expected_restricted = roc_auc_score(
            y_true,
            data.restricted_scores[:, phenotype_index],
            sample_weight=weights,
        )
        maximum_error = max(
            maximum_error,
            abs(
                expected_restricted
                - restricted_bootstrap[bootstrap_index, phenotype_index]
            ),
        )
        expected_full = np.mean(
            [
                roc_auc_score(
                    y_true,
                    data.full_scores[fit_index, :, phenotype_index],
                    sample_weight=weights,
                )
                for fit_index in range(N_FITS)
            ]
        )
        maximum_error = max(
            maximum_error,
            abs(expected_full - full_bootstrap[bootstrap_index, phenotype_index]),
        )
    if maximum_error > 1e-12:
        raise ValueError(
            "vectorized weighted AUROC validation failed: "
            f"maximum error={maximum_error}"
        )
    return full_bootstrap, restricted_bootstrap, float(maximum_error)


def write_tidy_csv(
    path: Path,
    data: AnalysisData,
    full_bootstrap: np.ndarray,
    restricted_bootstrap: np.ndarray,
    ci_level: float,
) -> None:
    fieldnames = [
        "phecode",
        "phenotype",
        "source_phenotype",
        "n_rule_retained_observations",
        "n_test_patients",
        "n_test_scans",
        "n_positive_test_scans",
        "full_bank_auc_mean_across_20_fits",
        "full_bank_ci_low",
        "full_bank_ci_high",
        "full_bank_n_bootstrap_valid",
        "rule_restricted_auc",
        "rule_restricted_ci_low",
        "rule_restricted_ci_high",
        "rule_restricted_n_bootstrap_valid",
        "direction",
        "ci_level",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index in range(EXPECTED_N_PHENOTYPES):
            full_low, full_high, full_valid = percentile_interval(
                full_bootstrap[:, index],
                ci_level,
            )
            restricted_low, restricted_high, restricted_valid = percentile_interval(
                restricted_bootstrap[:, index],
                ci_level,
            )
            direction = (
                "higher"
                if data.restricted_point_reference[index]
                > data.full_point_reference[index]
                else "lower"
            )
            writer.writerow(
                {
                    "phecode": data.phecodes[index],
                    "phenotype": data.display_phenotypes[index],
                    "source_phenotype": data.source_phenotypes[index],
                    "n_rule_retained_observations": int(
                        data.n_retained_observations[index]
                    ),
                    "n_test_patients": len(np.unique(data.person_ids)),
                    "n_test_scans": len(data.person_ids),
                    "n_positive_test_scans": int(data.labels[:, index].sum()),
                    "full_bank_auc_mean_across_20_fits": (
                        f"{data.full_point_reference[index]:.15g}"
                    ),
                    "full_bank_ci_low": f"{full_low:.15g}",
                    "full_bank_ci_high": f"{full_high:.15g}",
                    "full_bank_n_bootstrap_valid": full_valid,
                    "rule_restricted_auc": (
                        f"{data.restricted_point_reference[index]:.15g}"
                    ),
                    "rule_restricted_ci_low": f"{restricted_low:.15g}",
                    "rule_restricted_ci_high": f"{restricted_high:.15g}",
                    "rule_restricted_n_bootstrap_valid": restricted_valid,
                    "direction": direction,
                    "ci_level": f"{ci_level:.15g}",
                }
            )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.ci_level < 1.0:
        raise SystemExit("--ci-level must lie strictly between zero and one")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading and validating fixed predictions", flush=True)
    data = load_data(args)
    print(
        f"  aligned {len(data.person_ids):,} scans from "
        f"{len(np.unique(data.person_ids)):,} patients",
        flush=True,
    )
    print("[2/4] Building shared patient-cluster resamples", flush=True)
    volume_weights = build_shared_cluster_weights(
        data.person_ids,
        args.n_bootstrap,
        args.seed,
    )
    print("[3/4] Computing bootstrap AUROCs", flush=True)
    full_bootstrap, restricted_bootstrap, implementation_error = run_bootstrap(
        data,
        volume_weights,
    )

    tidy_path = output_dir / "fig6_all86_patient_cluster_bootstrap_tidy.csv"
    bootstrap_path = output_dir / "fig6_all86_patient_cluster_bootstrap.npz"
    status_path = output_dir / "fig6_all86_patient_cluster_bootstrap_status.json"
    print("[4/4] Writing retained analysis outputs", flush=True)
    write_tidy_csv(
        tidy_path,
        data,
        full_bootstrap,
        restricted_bootstrap,
        args.ci_level,
    )
    np.savez_compressed(
        bootstrap_path,
        phecodes=np.asarray(data.phecodes, dtype=object),
        phenotypes=np.asarray(data.display_phenotypes, dtype=object),
        full_bank_mean_across_20_fits_auroc=full_bootstrap.astype(np.float32),
        rule_restricted_auroc=restricted_bootstrap.astype(np.float32),
        n_bootstrap=np.asarray(args.n_bootstrap),
        seed=np.asarray(args.seed),
        ci_level=np.asarray(args.ci_level),
        n_test_patients=np.asarray(len(np.unique(data.person_ids))),
        n_test_scans=np.asarray(len(data.person_ids)),
    )

    full_macro = full_bootstrap.mean(axis=1)
    restricted_macro = restricted_bootstrap.mean(axis=1)
    full_macro_low, full_macro_high, full_macro_valid = percentile_interval(
        full_macro,
        args.ci_level,
    )
    restricted_macro_low, restricted_macro_high, restricted_macro_valid = (
        percentile_interval(restricted_macro, args.ci_level)
    )
    higher = int(
        np.sum(data.restricted_point_reference > data.full_point_reference)
    )
    lower = int(
        np.sum(data.restricted_point_reference < data.full_point_reference)
    )
    if (higher, lower) != (EXPECTED_HIGHER, EXPECTED_LOWER):
        raise ValueError(f"direction counts changed: higher={higher}, lower={lower}")
    status = {
        "schema_version": "fig6-all86-patient-cluster-bootstrap-v1",
        "status": "complete",
        "method": {
            "confidence_level": args.ci_level,
            "interval": "two-sided percentile interval",
            "n_resamples": args.n_bootstrap,
            "seed": args.seed,
            "sampling_unit": "INSPECT person_id",
            "shared_patient_resamples_across_both_profiles_and_all_phenotypes": True,
            "all_scans_from_each_sampled_patient_receive_the_same_multiplicity": True,
            "models_refit_within_bootstrap": False,
            "full_bank_estimand": (
                "mean held-out AUROC across 20 fixed validation-selected Adam "
                "probe fits; AUROCs are averaged within each patient resample"
            ),
            "rule_restricted_estimand": (
                "held-out AUROC of the fixed fitted rule-restricted probe"
            ),
        },
        "cohort": {
            "n_test_patients": len(np.unique(data.person_ids)),
            "n_test_scans": len(data.person_ids),
            "n_phenotypes": EXPECTED_N_PHENOTYPES,
        },
        "summary": {
            "full_bank_mean_auc": float(data.full_point_reference.mean()),
            "full_bank_macro_ci_low": full_macro_low,
            "full_bank_macro_ci_high": full_macro_high,
            "full_bank_macro_n_bootstrap_valid": full_macro_valid,
            "rule_restricted_mean_auc": float(
                data.restricted_point_reference.mean()
            ),
            "rule_restricted_macro_ci_low": restricted_macro_low,
            "rule_restricted_macro_ci_high": restricted_macro_high,
            "rule_restricted_macro_n_bootstrap_valid": restricted_macro_valid,
            "n_higher_rule_restricted": higher,
            "n_lower_rule_restricted": lower,
            "n_unchanged": EXPECTED_N_PHENOTYPES - higher - lower,
        },
        "validation": {
            "all_bootstrap_aurocs_finite": True,
            "maximum_weighted_auc_implementation_error": implementation_error,
            "maximum_full_bank_per_fit_auc_reconstruction_error": float(
                np.max(
                    np.abs(
                        data.full_per_fit_reconstructed
                        - data.full_per_fit_reference
                    )
                )
            ),
            "maximum_full_bank_mean_auc_reconstruction_error": float(
                np.max(
                    np.abs(
                        data.full_point_reconstructed
                        - data.full_point_reference
                    )
                )
            ),
            "maximum_rule_restricted_auc_reconstruction_error": float(
                np.max(
                    np.abs(
                        data.restricted_point_reconstructed
                        - data.restricted_point_reference
                    )
                )
            ),
            "aggregate_point_estimates_match_figure6_ledger": (
                abs(
                    float(data.full_point_reference.mean())
                    - EXPECTED_FULL_MEAN
                )
                <= AGGREGATE_TOLERANCE
                and abs(
                    float(data.restricted_point_reference.mean())
                    - EXPECTED_RESTRICTED_MEAN
                )
                <= AGGREGATE_TOLERANCE
            ),
            "direction_counts_match_figure6_ledger": True,
        },
        "inputs": [
            {
                "path": relative_to_root(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in data.input_paths
        ],
        "outputs": {
            "tidy_csv": {
                "path": relative_to_root(tidy_path),
                "size_bytes": tidy_path.stat().st_size,
                "sha256": sha256(tidy_path),
            },
            "bootstrap_distributions": {
                "path": relative_to_root(bootstrap_path),
                "size_bytes": bootstrap_path.stat().st_size,
                "sha256": sha256(bootstrap_path),
            },
        },
        "limitations": [
            (
                "Intervals capture held-out patient-cluster sampling "
                "variability conditional on the retained fitted probes; "
                "models are not refit within resamples."
            ),
            (
                "The full-bank point estimate and each full-bank bootstrap "
                "replicate average AUROCs across the same 20 fixed fits."
            ),
            (
                "The comparison remains descriptive; no phenotype-level or "
                "aggregate hypothesis tests are reported."
            ),
        ],
    }
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Figure 6 patient-cluster bootstrap complete: "
        f"full={data.full_point_reference.mean():.6f} "
        f"({full_macro_low:.6f}, {full_macro_high:.6f}); "
        f"restricted={data.restricted_point_reference.mean():.6f} "
        f"({restricted_macro_low:.6f}, {restricted_macro_high:.6f})"
    )
    print(f"  -> {tidy_path}")
    print(f"  -> {bootstrap_path}")
    print(f"  -> {status_path}")


if __name__ == "__main__":
    main()

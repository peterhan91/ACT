#!/usr/bin/env python3
"""Reconstruct and validate the row-aligned RSNA-2023 test manifest.

Only metadata are required.  The original pipeline repurposed the Kaggle
training cohort, split patients 70/10/20 with seed 42, and wrote each split in
lexicographic NIfTI-path order.  The public competition ``train.csv`` predates
the corrected ``train_2024.csv`` used by the pipeline, so it is used here only
to audit label-version differences.  The exact retained 943 x 9 outcome matrix
is recovered from the fair-comparison NPZ and is never inferred from model
predictions.

The script fails unless all patient-cluster and row-alignment gates pass:

* the reconstructed test split has 629 patients and 943 unique series;
* its lexicographic series-order checksum matches the pinned deterministic
  reconstruction;
* every reconstructed patient has one constant retained outcome vector;
* all retained fair-comparison NPZs agree on labels, outcomes and h5 indices;
* the reconstructed outcomes reproduce all 54 displayed zero-shot AUROCs.

The downloaded Kaggle metadata remain below the git-ignored ``_local_data/``
directory.  The generated analysis manifest replaces patient and series IDs
with sequential row and cluster identifiers, so it contains no linkable source
identifiers and can safely live beside the non-identifying validation summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP1 = ROOT / "experiments" / "exp1_zeroshot"
LOCAL_DATA = HERE / "_local_data" / "rsna2023"

EXPECTED_COMPETITION_LABELS_SHA256 = (
    "90c66499ac0f4f34a45073f90c62e34b790d05f854be4913d2308f6739368453"
)
EXPECTED_SERIES_META_SHA256 = (
    "9bee2a47fa3b0d1af904bcf565104b89ede0838217a5a1c7f401af285208039f"
)
EXPECTED_TEST_ORDER_SHA256 = (
    "3aa10409dc35ef855762a1b5803d166ff3896cb2a5454443a03a937cf348e8f2"
)
EXPECTED_Y_SHA256 = (
    "02889610c61458932b5e983b01b411018ff343f7c41d876e4a4b1873572f9f66"
)
EXPECTED_PATIENT_CLUSTER_SHA256 = (
    "ff918958cdeb0e8367576d6bd81c358b449ff57fb2623cea8743d9f87a992645"
)
EXPECTED_LABELS = [
    "bowel_injury",
    "extravasation_injury",
    "kidney_low",
    "kidney_high",
    "liver_low",
    "liver_high",
    "spleen_low",
    "spleen_high",
    "any_injury",
]
DISPLAYED_ZERO_SHOT_MODELS = {
    "ours": "plain",
    "merlin": "native",
    "ctclip": "native",
    "medsiglip": "native",
    "biomedclip": "native",
    "openai_clip": "native",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--competition-labels",
        type=Path,
        default=LOCAL_DATA / "train.csv",
        help="Original Kaggle train.csv; used only for a label-version audit.",
    )
    parser.add_argument(
        "--series-meta",
        type=Path,
        default=LOCAL_DATA / "train_series_meta.csv",
    )
    parser.add_argument(
        "--retained-labels-npz",
        type=Path,
        default=(
            ROOT
            / "outputs/v1/probe_compare/fairA/perseed/rsna__f2llm.npz"
        ),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=HERE / "outputs" / "rsna2023_test_manifest.csv",
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=HERE / "outputs" / "rsna2023_manifest_validation.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios", nargs=3, type=float, default=[0.7, 0.1, 0.2]
    )
    parser.add_argument("--auc-tol", type=float, default=1e-12)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _auc_from_sorted_groups(y: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUROC with exact mid-rank treatment of tied scores."""
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores):
        raise ValueError(f"AUROC shape mismatch: y={y.shape}, scores={scores.shape}")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("AUROC inputs must contain finite scores and binary labels")
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])]
    group_pos = np.add.reduceat(sorted_y.astype(np.float64), starts)
    group_neg = np.add.reduceat((1 - sorted_y).astype(np.float64), starts)
    negative_before = np.cumsum(group_neg) - group_neg
    numerator = np.sum(group_pos * (negative_before + 0.5 * group_neg))
    return float(numerator / (n_pos * n_neg))


def _as_identifier(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("patient_id/series_id contains a non-integral value")
    return numeric.astype(np.int64).astype(str)


def _load_retained_outcomes(
    reference_npz: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    with np.load(reference_npz, allow_pickle=False) as payload:
        y = np.asarray(payload["y"], dtype=np.int8)
        h5_idx = np.asarray(payload["h5_idx"], dtype=np.int64)
        labels = [str(value) for value in payload["labels"].tolist()]
    if labels != EXPECTED_LABELS:
        raise ValueError(f"retained label order changed: {labels}")
    if y.shape != (943, len(EXPECTED_LABELS)):
        raise ValueError(f"unexpected retained y shape: {y.shape}")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("retained outcomes are not binary")
    if not np.array_equal(h5_idx, np.arange(len(y), dtype=np.int64)):
        raise ValueError("retained h5_idx is not the exact row range 0..942")
    if sha256_array(y) != EXPECTED_Y_SHA256:
        raise ValueError("retained outcome checksum changed")

    peer_summaries: dict[str, Any] = {}
    peer_paths = sorted(reference_npz.parent.glob("rsna__*.npz"))
    for path in peer_paths:
        with np.load(path, allow_pickle=False) as peer:
            peer_y = np.asarray(peer["y"], dtype=np.int8)
            peer_h5 = np.asarray(peer["h5_idx"], dtype=np.int64)
            peer_labels = [str(value) for value in peer["labels"].tolist()]
        same = bool(
            np.array_equal(peer_y, y)
            and np.array_equal(peer_h5, h5_idx)
            and peer_labels == labels
        )
        if not same:
            raise ValueError(f"retained RSNA alignment differs in {path}")
        peer_summaries[str(path.relative_to(ROOT))] = {
            "file_sha256": sha256_file(path),
            "y_sha256": sha256_array(peer_y),
        }
    return y, h5_idx, labels, peer_summaries


def _reconstruct_test_rows(
    series_meta: pd.DataFrame,
    seed: int,
    ratios: tuple[float, float, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"split ratios must sum to one: {ratios}")
    required = {"patient_id", "series_id"}
    if missing := required.difference(series_meta.columns):
        raise ValueError(f"series metadata missing columns: {sorted(missing)}")
    if len(series_meta) != 4711:
        raise ValueError(f"expected 4,711 series, found {len(series_meta):,}")

    frame = series_meta.copy()
    frame["patient_id"] = _as_identifier(frame["patient_id"])
    frame["series_id"] = _as_identifier(frame["series_id"])
    if frame[["patient_id", "series_id"]].duplicated().any():
        raise ValueError("series metadata contains duplicate patient/series pairs")
    frame["VolumeName"] = (
        frame["patient_id"] + "__" + frame["series_id"] + ".nii.gz"
    )
    if frame["VolumeName"].duplicated().any():
        raise ValueError("reconstructed volume names are not unique")
    frame = frame.sort_values("VolumeName", kind="stable").reset_index(drop=True)

    patient_ids = sorted(frame["patient_id"].drop_duplicates().tolist())
    if len(patient_ids) != 3147:
        raise ValueError(f"expected 3,147 patients, found {len(patient_ids):,}")
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)
    n_train = int(round(len(patient_ids) * ratios[0]))
    n_valid = int(round(len(patient_ids) * ratios[1]))
    split_ids = {
        "train": set(patient_ids[:n_train]),
        "valid": set(patient_ids[n_train : n_train + n_valid]),
        "test": set(patient_ids[n_train + n_valid :]),
    }
    split_counts = {
        name: {
            "n_patients": len(ids),
            "n_series": int(frame["patient_id"].isin(ids).sum()),
        }
        for name, ids in split_ids.items()
    }
    expected_counts = {
        "train": {"n_patients": 2203, "n_series": 3284},
        "valid": {"n_patients": 315, "n_series": 484},
        "test": {"n_patients": 629, "n_series": 943},
    }
    if split_counts != expected_counts:
        raise ValueError(
            f"reconstructed split counts changed: {split_counts} != {expected_counts}"
        )

    test = (
        frame[frame["patient_id"].isin(split_ids["test"])]
        .sort_values("VolumeName", kind="stable")
        .reset_index(drop=True)
    )
    order_text = "".join(f"{name}\n" for name in test["VolumeName"])
    order_sha256 = hashlib.sha256(order_text.encode("utf-8")).hexdigest()
    if order_sha256 != EXPECTED_TEST_ORDER_SHA256:
        raise ValueError(f"test series-order checksum changed: {order_sha256}")
    test.insert(1, "h5_idx", np.arange(len(test), dtype=np.int64))
    return test, {
        "algorithm": (
            "lexicographically sort string patient IDs; shuffle with "
            "numpy.default_rng(seed); round 70% and 10% cut points; assign the "
            "remainder to test; lexicographically sort patient__series filenames"
        ),
        "seed": seed,
        "ratios": list(ratios),
        "split_counts": split_counts,
        "test_series_order_sha256": order_sha256,
    }


def _validate_zero_shot_aurocs(
    y: np.ndarray, labels: list[str], tolerance: float
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    all_differences: list[float] = []
    for model, mode in DISPLAYED_ZERO_SHOT_MODELS.items():
        result_path = EXP1 / model / "results" / f"rsna2023_test__{mode}__results.json"
        probs_path = result_path.with_name(f"rsna2023_test__{mode}__probs.npy")
        result = json.loads(result_path.read_text())
        result_labels = [str(value) for value in result["labels"]]
        probs = np.load(probs_path, mmap_mode="r")
        if probs.shape != (len(y), len(result_labels)):
            raise ValueError(f"{model}: unexpected prediction shape {probs.shape}")
        column = {label: i for i, label in enumerate(result_labels)}
        differences: dict[str, float] = {}
        for label_index, label in enumerate(labels):
            point = _auc_from_sorted_groups(y[:, label_index], probs[:, column[label]])
            expected = float(result["per_label_auc"][label])
            difference = abs(point - expected)
            differences[label] = difference
            all_differences.append(difference)
        model_max = max(differences.values())
        if model_max > tolerance:
            raise ValueError(
                f"{model}: AUROC reproduction failed; max difference={model_max:.3g}"
            )
        models[model] = {
            "mode": mode,
            "n_label_matches": len(differences),
            "max_absolute_difference": model_max,
            "results_json": str(result_path.relative_to(ROOT)),
            "probs_npy": str(probs_path.relative_to(ROOT)),
        }
    return {
        "n_models": len(models),
        "n_model_label_matches": len(all_differences),
        "tolerance": tolerance,
        "max_absolute_difference": max(all_differences),
        "models": models,
    }


def main() -> None:
    args = parse_args()
    labels_sha256 = sha256_file(args.competition_labels)
    series_sha256 = sha256_file(args.series_meta)
    if labels_sha256 != EXPECTED_COMPETITION_LABELS_SHA256:
        raise ValueError(f"competition train.csv checksum changed: {labels_sha256}")
    if series_sha256 != EXPECTED_SERIES_META_SHA256:
        raise ValueError(f"train_series_meta.csv checksum changed: {series_sha256}")

    competition_labels = pd.read_csv(args.competition_labels)
    series_meta = pd.read_csv(args.series_meta)
    if len(competition_labels) != 3147 or competition_labels["patient_id"].duplicated().any():
        raise ValueError("competition train.csv is not one row per 3,147 patients")
    test, split_summary = _reconstruct_test_rows(
        series_meta, args.seed, tuple(args.ratios)
    )
    y, h5_idx, labels, peer_summaries = _load_retained_outcomes(
        args.retained_labels_npz
    )
    if not np.array_equal(test["h5_idx"].to_numpy(), h5_idx):
        raise ValueError("reconstructed rows do not align with retained h5_idx")

    patient_cluster, unique_patients = pd.factorize(test["patient_id"], sort=False)
    if len(unique_patients) != 629 or (patient_cluster < 0).any():
        raise ValueError("failed to construct the expected 629 patient clusters")
    manifest = pd.DataFrame(
        {
            "VolumeName": [f"rsna2023_test_{index:04d}" for index in range(len(test))],
            "h5_idx": h5_idx,
            "patient_cluster": patient_cluster.astype(np.int64),
        }
    )
    patient_cluster_sha256 = sha256_array(
        manifest["patient_cluster"].to_numpy(dtype=np.int64)
    )
    if patient_cluster_sha256 != EXPECTED_PATIENT_CLUSTER_SHA256:
        raise ValueError("reconstructed patient-cluster checksum changed")
    for index, label in enumerate(labels):
        manifest[label] = y[:, index]

    inconsistent_patients: list[str] = []
    for patient_id, indices in test.groupby("patient_id", sort=False).groups.items():
        rows = y[np.asarray(list(indices), dtype=np.int64)]
        if len(np.unique(rows, axis=0)) != 1:
            inconsistent_patients.append(str(patient_id))
    if inconsistent_patients:
        raise ValueError(
            "retained outcomes are not constant within reconstructed patient groups: "
            f"{inconsistent_patients[:10]}"
        )

    old = competition_labels.copy()
    old["patient_id"] = _as_identifier(old["patient_id"])
    corrected_rows = test[["patient_id"]].copy()
    for index, label in enumerate(labels):
        corrected_rows[label] = y[:, index]
    corrected_patient = corrected_rows.drop_duplicates("patient_id", keep="first")
    comparison = corrected_patient[["patient_id"] + labels].merge(
        old[["patient_id"] + labels],
        on="patient_id",
        how="left",
        validate="one_to_one",
        suffixes=("_retained", "_competition"),
    )
    changed = np.column_stack(
        [
            comparison[f"{label}_retained"].to_numpy()
            != comparison[f"{label}_competition"].to_numpy()
            for label in labels
        ]
    )
    label_version_audit = {
        "original_competition_labels_used_for_outcomes": False,
        "reason": (
            "The preprocessing pipeline used corrected train_2024.csv labels; "
            "the original competition train.csv is retained only for version auditing."
        ),
        "n_test_patients_with_any_changed_label": int(changed.any(axis=1).sum()),
        "n_changed_patient_label_cells": int(changed.sum()),
        "changed_patients_by_label": {
            label: int(changed[:, index].sum()) for index, label in enumerate(labels)
        },
    }

    auc_validation = _validate_zero_shot_aurocs(y, labels, args.auc_tol)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.status_json.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_manifest, index=False)

    patient_sizes = manifest.groupby("patient_cluster", sort=False).size()
    status = {
        "status": "complete",
        "purpose": "row-aligned patient-cluster manifest for RSNA-2023 test AUROC CIs",
        "metadata_sources": {
            "competition_page": (
                "https://www.kaggle.com/competitions/"
                "rsna-2023-abdominal-trauma-detection/data"
            ),
            "competition_labels": {
                "path": str(args.competition_labels.relative_to(ROOT)),
                "sha256": labels_sha256,
            },
            "series_meta": {
                "path": str(args.series_meta.relative_to(ROOT)),
                "sha256": series_sha256,
            },
            "retained_outcomes_npz": {
                "path": str(args.retained_labels_npz.relative_to(ROOT)),
                "file_sha256": sha256_file(args.retained_labels_npz),
                "y_sha256": sha256_array(y),
            },
        },
        "split_reconstruction": split_summary,
        "mapping_provenance": {
            "status": "reconstructed_from_recorded_seed_and_pipeline",
            "original_splitter_retained": False,
            "original_split_csv_retained": False,
            "interpretation": (
                "The checksum is a regression guard for the deterministic "
                "reconstruction. AUROC reproduction validates outcome and "
                "prediction row alignment, not patient identity by itself."
            ),
        },
        "row_alignment_gates": {
            "n_test_series": len(manifest),
            "n_test_patients": int(manifest["patient_cluster"].nunique()),
            "n_multiseries_patients": int((patient_sizes > 1).sum()),
            "all_629_patient_groups_have_constant_retained_outcomes": True,
            "h5_idx_is_exact_range_0_to_942": True,
            "patient_cluster_sha256": patient_cluster_sha256,
            "n_identical_fair_comparison_npz_archives": len(peer_summaries),
            "fair_comparison_npz_archives": peer_summaries,
        },
        "label_version_audit": label_version_audit,
        "zero_shot_auc_validation": auc_validation,
        "output_manifest": {
            "path": str(args.output_manifest.relative_to(ROOT)),
            "sha256": sha256_file(args.output_manifest),
            "contains_source_patient_or_series_identifiers": False,
        },
    }
    args.status_json.write_text(json.dumps(status, indent=2) + "\n")
    print(
        "[complete] RSNA-2023 manifest: "
        f"{len(manifest)} series, {manifest['patient_cluster'].nunique()} patients; "
        f"54/54 zero-shot AUROCs reproduced; wrote {args.output_manifest}",
        flush=True,
    )


if __name__ == "__main__":
    main()

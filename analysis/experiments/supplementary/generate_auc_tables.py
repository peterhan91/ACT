#!/usr/bin/env python3
"""Generate reproducible supplementary AUROC tables and LaTeX fragments.

Outputs
-------
1. Zero-shot finding annotation on CT-RATE test, PMBB chest non-contrast,
   PMBB abdomen contrast and RSNA-2023 abdominal trauma. Confidence intervals
   are 95% percentile intervals from shared patient-cluster bootstrap
   resamples. All scans from a sampled patient are retained; PMBB has one
   selected scan per patient.
2. The complete 221-phenotype matched linear-probe comparison between ACT
   and CT-CLIP. Confidence intervals are two-sided t intervals across the 20
   matched probe fits and therefore quantify fit variation, not patient
   sampling uncertainty.
3. A machine-readable status/validation record with input, alignment and
   resampling provenance.

The script is intentionally independent of the manuscript repository. Run it
from anywhere; paths are resolved relative to this repository checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP1 = ROOT / "experiments" / "exp1_zeroshot"

MODEL_ORDER = [
    "ours",
    "merlin",
    "ctclip",
    "medsiglip",
    "biomedclip",
    "openai_clip",
]
MODEL_SPECS = {
    "ours": {"display": "ACT", "mode": "plain"},
    "merlin": {"display": "Merlin", "mode": "native"},
    "ctclip": {"display": "CT-CLIP", "mode": "native"},
    "fvlm": {"display": "f-VLM", "mode": "native"},
    "medsiglip": {"display": "MedSigLIP", "mode": "native"},
    "biomedclip": {"display": "BiomedCLIP", "mode": "native"},
    "openai_clip": {"display": "OpenAI CLIP", "mode": "native"},
}


@dataclass(frozen=True)
class DatasetSpec:
    display: str
    labels_csv: Path
    manifest_csv: Path
    patient_column: str | None
    patient_from_volume_regex: str | None = None
    validation_json: Path | None = None


DATASETS = {
    "ctrate_test": DatasetSpec(
        display="CT-RATE test",
        labels_csv=ROOT / "outputs/v1/cache/volume_index.ctrate_test.csv",
        manifest_csv=ROOT / "outputs/v1/cache/volume_index.ctrate_test.csv",
        patient_column=None,
        patient_from_volume_regex=r"^(valid_\d+)_",
    ),
    "pmbb_chest_nc": DatasetSpec(
        display="PMBB chest non-contrast",
        labels_csv=EXP1 / "pmbb_labels/labels/pmbb_chest_nc_labels.csv",
        manifest_csv=ROOT / "experiments/exp2_retrieval/pmbb_manifests/pmbb_chest_nc.csv",
        patient_column="patient",
    ),
    "pmbb_abd_ce": DatasetSpec(
        display="PMBB abdomen contrast-enhanced",
        labels_csv=EXP1 / "pmbb_labels/labels/pmbb_abd_ce_labels.csv",
        manifest_csv=ROOT / "experiments/exp2_retrieval/pmbb_manifests/pmbb_abd_ce.csv",
        patient_column="patient",
    ),
    "rsna2023_test": DatasetSpec(
        display="RSNA-2023 abdominal trauma",
        labels_csv=HERE / "outputs/rsna2023_test_manifest.csv",
        manifest_csv=HERE / "outputs/rsna2023_test_manifest.csv",
        patient_column="patient_cluster",
        validation_json=HERE / "outputs/rsna2023_manifest_validation.json",
    ),
}

SECTOR_ORDER = [
    "Infectious",
    "Neoplasms",
    "Hematopoietic",
    "Endocrine/Metabolic",
    "Circulatory",
    "Sepsis/SIRS",
    "Genitourinary",
    "Respiratory",
    "Digestive",
    "Mental",
    "Neurological/Sense",
    "Musculoskeletal",
    "Dermatologic",
    "Congenital",
    "Pregnancy",
    "Symptoms",
    "Injury",
    "Other/Admin",
]
SECTOR_OVERRIDE = {
    771.1: "Symptoms",
    1013.0: "Symptoms",
    994.1: "Sepsis/SIRS",
    994.2: "Sepsis/SIRS",
    994.21: "Sepsis/SIRS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--bootstrap-chunk", type=int, default=100)
    parser.add_argument("--validation-tol", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    return parser.parse_args()


def _json_auc_map(payload: dict[str, Any]) -> dict[str, float]:
    values = payload["per_label_auc"]
    if isinstance(values, dict):
        return {str(k): float(v) for k, v in values.items()}
    return {str(k): float(v) for k, v in zip(payload["labels"], values)}


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _auc_from_sorted_groups(y: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUROC with exact mid-rank handling for tied prediction scores."""
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores):
        raise ValueError(f"AUROC shape mismatch: y={y.shape}, scores={scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("AUROC scores contain NaN or infinity")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("AUROC labels must be binary 0/1")
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


def _shared_cluster_draws(
    n_clusters: int, n_boot: int, seed: int, dataset_index: int
) -> tuple[np.ndarray, str]:
    """Return B x C cluster multiplicities and a content hash for provenance."""
    if n_clusters > np.iinfo(np.uint16).max:
        raise ValueError("uint16 bootstrap counts cannot represent this cluster count")
    rng = np.random.default_rng(np.random.SeedSequence([seed, dataset_index]))
    counts = np.empty((n_boot, n_clusters), dtype=np.uint16)
    for b in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        counts[b] = np.bincount(draw, minlength=n_clusters)
    digest = hashlib.sha256(counts.tobytes(order="C")).hexdigest()
    return counts, digest


def _cluster_bootstrap_auc(
    y: np.ndarray,
    scores: np.ndarray,
    row_cluster: np.ndarray,
    cluster_counts: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Weighted AUROCs for shared cluster draws, sorting the scores only once."""
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    row_cluster = np.asarray(row_cluster, dtype=np.int64)
    if not (len(y) == len(scores) == len(row_cluster)):
        raise ValueError("bootstrap y/scores/cluster arrays are not row-aligned")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order].astype(np.float64)
    sorted_cluster = row_cluster[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1])]

    out = np.full(cluster_counts.shape[0], np.nan, dtype=np.float64)
    for lo in range(0, len(out), chunk_size):
        hi = min(lo + chunk_size, len(out))
        weights = cluster_counts[lo:hi, sorted_cluster].astype(np.float64)
        group_pos = np.add.reduceat(weights * sorted_y, starts, axis=1)
        group_neg = np.add.reduceat(weights * (1.0 - sorted_y), starts, axis=1)
        negative_before = np.cumsum(group_neg, axis=1) - group_neg
        numerator = np.sum(group_pos * (negative_before + 0.5 * group_neg), axis=1)
        denominator = group_pos.sum(axis=1) * group_neg.sum(axis=1)
        np.divide(numerator, denominator, out=out[lo:hi], where=denominator > 0)
    return out


def _load_zero_shot_dataset(
    key: str, spec: DatasetSpec
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, pd.DataFrame]:
    if spec.validation_json is not None:
        if not spec.validation_json.exists():
            raise FileNotFoundError(
                f"{key}: missing reconstruction validation {spec.validation_json}"
            )
        validation = json.loads(spec.validation_json.read_text())
        if validation.get("status") != "complete":
            raise ValueError(f"{key}: reconstruction validation is not complete")
        expected_sha256 = validation["output_manifest"]["sha256"]
        observed_sha256 = _file_sha256(spec.manifest_csv)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"{key}: manifest checksum differs from reconstruction validation"
            )
        gates = validation["row_alignment_gates"]
        if not (
            gates["n_test_series"] == 943
            and gates["n_test_patients"] == 629
            and gates["all_629_patient_groups_have_constant_retained_outcomes"]
            and gates["h5_idx_is_exact_range_0_to_942"]
        ):
            raise ValueError(f"{key}: a required patient-row alignment gate failed")
        auc_gate = validation["zero_shot_auc_validation"]
        if auc_gate["n_model_label_matches"] != 54:
            raise ValueError(f"{key}: expected 54 validated displayed-model AUROCs")

    labels = pd.read_csv(spec.labels_csv)
    manifest = pd.read_csv(spec.manifest_csv)
    if labels["VolumeName"].duplicated().any() or manifest["VolumeName"].duplicated().any():
        raise ValueError(f"{key}: VolumeName must be unique in labels and manifest")

    if spec.labels_csv.resolve() == spec.manifest_csv.resolve():
        merged = labels.copy()
    else:
        patient_cols = ["VolumeName"] + ([spec.patient_column] if spec.patient_column else [])
        merged = manifest[patient_cols].merge(
            labels, on="VolumeName", how="left", validate="one_to_one", sort=False
        )
        if not merged["VolumeName"].equals(manifest["VolumeName"]):
            raise ValueError(f"{key}: merge changed canonical manifest row order")

    canonical_table = pd.read_csv(
        EXP1 / f"analysis/tables/per_label_auc__{key}.csv"
    )
    label_order = canonical_table["label"].astype(str).tolist()
    missing = [label for label in label_order if label not in merged.columns]
    if missing:
        raise ValueError(f"{key}: labels absent from source matrix: {missing}")
    y = merged[label_order].to_numpy(dtype=np.int8)
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{key}: source labels are not binary")

    if spec.patient_column:
        patient = merged[spec.patient_column].astype(str)
    else:
        patient = merged["VolumeName"].str.extract(spec.patient_from_volume_regex)[0]
        if patient.isna().any():
            raise ValueError(f"{key}: patient regex failed for {int(patient.isna().sum())} rows")
    row_cluster, unique_patients = pd.factorize(patient, sort=False)
    if (row_cluster < 0).any():
        raise ValueError(f"{key}: missing patient identifiers")

    if "h5_idx" in merged.columns:
        idx = merged["h5_idx"].to_numpy(dtype=np.int64)
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"{key}: h5_idx is not unique")
    return merged, label_order, y, row_cluster, canonical_table


def generate_zero_shot(
    n_boot: int,
    seed: int,
    chunk_size: int,
    validation_tol: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset_status: dict[str, Any] = {}
    discrepancies: list[dict[str, Any]] = []

    for dataset_index, (dataset, spec) in enumerate(DATASETS.items()):
        merged, label_order, y_all, row_cluster, canonical_table = _load_zero_shot_dataset(
            dataset, spec
        )
        n_clusters = int(row_cluster.max() + 1)
        cluster_counts, resample_hash = _shared_cluster_draws(
            n_clusters, n_boot, seed, dataset_index
        )
        display_map = canonical_table.set_index("label")["display"].astype(str).to_dict()
        group_map = canonical_table.set_index("label")["group"].astype(str).to_dict()
        y_by_label = {label: y_all[:, i] for i, label in enumerate(label_order)}
        exceptions: list[dict[str, Any]] = []
        complete_models: list[str] = []

        for model in MODEL_ORDER:
            model_spec = MODEL_SPECS[model]
            result_path = (
                EXP1
                / model
                / "results"
                / f"{dataset}__{model_spec['mode']}__results.json"
            )
            probs_path = result_path.with_name(
                f"{dataset}__{model_spec['mode']}__probs.npy"
            )
            payload = json.loads(result_path.read_text()) if result_path.exists() else None
            auc_map = _json_auc_map(payload) if payload else {}
            json_labels = [str(x) for x in payload["labels"]] if payload else []
            probs = np.load(probs_path, mmap_mode="r") if probs_path.exists() else None

            aligned = bool(
                payload
                and probs is not None
                and probs.shape == (len(merged), len(json_labels))
                and int(payload["n_volumes"]) == len(merged)
            )
            if aligned:
                complete_models.append(model)
            elif payload:
                reason = (
                    "retained_prediction_rows_lack_case_identifiers"
                    if probs is not None and probs.shape[0] != len(merged)
                    else "prediction_file_missing_or_shape_mismatch"
                )
                exceptions.append(
                    {
                        "model": model,
                        "reason_code": reason,
                        "canonical_n_scans": int(payload["n_volumes"]),
                        "full_cohort_n_scans": int(len(merged)),
                        "prediction_shape": list(probs.shape) if probs is not None else None,
                    }
                )
            else:
                exceptions.append(
                    {"model": model, "reason_code": "model_not_applicable_to_dataset"}
                )

            json_col = {label: i for i, label in enumerate(json_labels)}
            for label in label_order:
                common = {
                    "dataset": dataset,
                    "dataset_display": spec.display,
                    "group": group_map[label],
                    "label": label,
                    "display_label": display_map[label],
                    "model": model,
                    "model_display": model_spec["display"],
                    "mode": model_spec["mode"],
                    "full_cohort_n_scans": int(len(merged)),
                    "full_cohort_n_patients": n_clusters,
                    "full_cohort_n_positive": int(y_by_label[label].sum()),
                    "full_cohort_n_negative": int(len(merged) - y_by_label[label].sum()),
                    "model_n_scans": int(payload["n_volumes"]) if payload else np.nan,
                    "model_n_patients": n_clusters if aligned else np.nan,
                    "model_n_positive": int(y_by_label[label].sum()) if aligned else np.nan,
                    "model_n_negative": int(len(merged) - y_by_label[label].sum()) if aligned else np.nan,
                    "ci_level": 0.95,
                    "ci_method": "patient_cluster_percentile_bootstrap" if aligned else "unavailable",
                    "n_boot": n_boot if aligned else 0,
                    "results_json": _relative(result_path) if result_path.exists() else "",
                    "probs_npy": _relative(probs_path) if probs_path.exists() else "",
                }

                if aligned and label in json_col:
                    scores = np.asarray(probs[:, json_col[label]], dtype=np.float64)
                    point = _auc_from_sorted_groups(y_by_label[label], scores)
                    canonical = float(auc_map[label])
                    delta = abs(point - canonical)
                    boot = _cluster_bootstrap_auc(
                        y_by_label[label], scores, row_cluster, cluster_counts, chunk_size
                    )
                    valid = boot[np.isfinite(boot)]
                    ci_lower, ci_upper = np.percentile(valid, [2.5, 97.5])
                    validation_status = "match" if delta <= validation_tol else "mismatch"
                    if validation_status == "mismatch":
                        discrepancies.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "label": label,
                                "recomputed_auc": point,
                                "canonical_auc": canonical,
                                "absolute_difference": delta,
                            }
                        )
                    rows.append(
                        {
                            **common,
                            "availability_status": "complete",
                            "auc": point,
                            "ci_lower": float(ci_lower),
                            "ci_upper": float(ci_upper),
                            "n_boot_valid": int(len(valid)),
                            "canonical_auc": canonical,
                            "validation_abs_difference": delta,
                            "validation_status": validation_status,
                        }
                    )
                elif payload and label in auc_map:
                    rows.append(
                        {
                            **common,
                            "availability_status": "point_only_row_mapping_missing",
                            "auc": float(auc_map[label]),
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                            "n_boot_valid": 0,
                            "canonical_auc": float(auc_map[label]),
                            "validation_abs_difference": np.nan,
                            "validation_status": "not_recomputed_unaligned_rows",
                        }
                    )
                else:
                    status = (
                        "not_applicable_label_not_covered"
                        if payload
                        else "not_applicable_model_dataset"
                    )
                    rows.append(
                        {
                            **common,
                            "availability_status": status,
                            "auc": np.nan,
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                            "n_boot_valid": 0,
                            "canonical_auc": np.nan,
                            "validation_abs_difference": np.nan,
                            "validation_status": "not_applicable",
                        }
                    )

            print(
                f"[{dataset}] {model}: "
                f"{'bootstrap complete' if aligned else 'point/status only'}",
                flush=True,
            )

        dataset_status[dataset] = {
            "status": "complete_with_documented_model_exceptions" if exceptions else "complete",
            "n_scans": int(len(merged)),
            "n_patients": n_clusters,
            "n_labels": len(label_order),
            "complete_ci_models": complete_models,
            "model_exceptions": exceptions,
            "bootstrap_resample_sha256": resample_hash,
            "labels_csv": _relative(spec.labels_csv),
            "manifest_csv": _relative(spec.manifest_csv),
            "validation_json": (
                _relative(spec.validation_json) if spec.validation_json else None
            ),
        }

    out = pd.DataFrame(rows)
    max_difference = float(out["validation_abs_difference"].max(skipna=True))
    status = {
        "bootstrap": {
            "method": "patient_cluster_percentile_bootstrap",
            "confidence_level": 0.95,
            "n_resamples": n_boot,
            "seed": seed,
            "shared_resamples_across_all_models_and_labels_within_dataset": True,
            "cluster_rule": (
                "sample patients with replacement and retain every scan belonging to each "
                "sampled patient; multiplicity is represented by integer observation weights"
            ),
            "percentile_method": "numpy_default_linear",
        },
        "datasets": dataset_status,
        "validation": {
            "target": "per-label AUROC in canonical results JSON",
            "tolerance": validation_tol,
            "max_absolute_difference": max_difference,
            "n_discrepancies": len(discrepancies),
            "discrepancies": discrepancies,
        },
        "model_scope": {
            "included": MODEL_ORDER,
            "excluded": {
                "m3dclip": (
                    "Excluded from the manuscript-facing model order and no restored "
                    "per-volume finding prediction arrays were available."
                )
            },
        },
    }
    return out, status


def _phecode_sector(phecode: str) -> str:
    """Mirror the reproducible PheWAS sector mapping used by plot_radar.py."""
    value = float(phecode)
    rounded = round(value, 2)
    if rounded in SECTOR_OVERRIDE:
        return SECTOR_OVERRIDE[rounded]
    for lower, upper, name in [
        (1, 140, "Infectious"),
        (140, 240, "Neoplasms"),
        (240, 280, "Endocrine/Metabolic"),
        (280, 290, "Hematopoietic"),
        (290, 320, "Mental"),
        (320, 390, "Neurological/Sense"),
        (390, 460, "Circulatory"),
        (460, 520, "Respiratory"),
        (520, 580, "Digestive"),
        (580, 630, "Genitourinary"),
        (630, 680, "Pregnancy"),
        (680, 710, "Dermatologic"),
        (710, 760, "Musculoskeletal"),
        (760, 780, "Congenital"),
        (780, 800, "Symptoms"),
        (800, 1000, "Injury"),
    ]:
        if lower <= value < upper:
            return name
    return "Other/Admin"


def generate_linear_probe() -> tuple[pd.DataFrame, dict[str, Any]]:
    perseed = ROOT / "outputs/v1/probe_compare/fairA/perseed"
    paths = {
        "f2llm": perseed / "inspect__f2llm.npz",
        "ctclip": perseed / "inspect__ctclip.npz",
    }
    archives = {name: np.load(path, allow_pickle=True) for name, path in paths.items()}
    f2llm = archives["f2llm"]
    ctclip = archives["ctclip"]

    for key in ["seeds", "labels", "y", "h5_idx"]:
        if not np.array_equal(f2llm[key], ctclip[key]):
            raise ValueError(f"linear probe archives differ in aligned field: {key}")
    seeds = np.asarray(f2llm["seeds"], dtype=np.int64)
    if len(seeds) != 20:
        raise ValueError(f"expected 20 matched seeds, found {len(seeds)}")

    labels = [str(x).strip() for x in f2llm["labels"]]
    y = np.asarray(f2llm["y"], dtype=np.int8)
    auc_f2llm = np.asarray(f2llm["per_label_auc"], dtype=np.float64)
    auc_ctclip = np.asarray(ctclip["per_label_auc"], dtype=np.float64)
    expected_shape = (len(seeds), len(labels))
    if auc_f2llm.shape != expected_shape or auc_ctclip.shape != expected_shape:
        raise ValueError(
            f"linear-probe AUROC shape mismatch: {auc_f2llm.shape}, {auc_ctclip.shape}, "
            f"expected {expected_shape}"
        )
    if y.shape != (2612, 221) or len(labels) != 221:
        raise ValueError(f"expected 2612 x 221 labels, found y={y.shape}, labels={len(labels)}")
    if not np.isfinite(auc_f2llm).all() or not np.isfinite(auc_ctclip).all():
        raise ValueError("linear-probe per-label AUROC arrays contain missing values")

    phecodes = pd.read_csv(EXP1 / "inspect_pheno/phecodes.csv", dtype=str)
    phecodes["phecode_str"] = phecodes["phecode_str"].str.strip()
    if labels != phecodes["phecode_str"].tolist():
        raise ValueError("linear-probe label order does not match phecodes.csv")

    t_critical = float(student_t.ppf(0.975, df=len(seeds) - 1))

    def summarize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        half_width = t_critical * sd / np.sqrt(len(seeds))
        return mean, sd, mean - half_width, mean + half_width

    f_mean, f_sd, f_lo, f_hi = summarize(auc_f2llm)
    c_mean, c_sd, c_lo, c_hi = summarize(auc_ctclip)
    delta_values = auc_f2llm - auc_ctclip
    d_mean, d_sd, d_lo, d_hi = summarize(delta_values)

    seed_macro_f = auc_f2llm.mean(axis=1)
    seed_macro_c = auc_ctclip.mean(axis=1)
    archived_macro_f = np.asarray(f2llm["test_mean_auc"], dtype=np.float64)
    archived_macro_c = np.asarray(ctclip["test_mean_auc"], dtype=np.float64)
    macro_validation = {
        "f2llm_max_abs_difference": float(np.max(np.abs(seed_macro_f - archived_macro_f))),
        "ctclip_max_abs_difference": float(np.max(np.abs(seed_macro_c - archived_macro_c))),
    }

    rows = []
    for j, label in enumerate(labels):
        phecode = str(phecodes.iloc[j]["phecode"]).strip()
        # Preserve the manuscript's canonical display for integer phecodes
        # while leaving genuinely hierarchical codes (for example 174.11)
        # unchanged.
        if phecode.endswith(".0") and phecode[:-2].isdigit():
            phecode = phecode[:-2]
        n_positive = int(y[:, j].sum())
        rows.append(
            {
                "sector": _phecode_sector(phecode),
                "phecode": phecode,
                "phenotype": label,
                "n_scans": int(len(y)),
                "n_positive": n_positive,
                "n_negative": int(len(y) - n_positive),
                "n_seeds": int(len(seeds)),
                "ci_level": 0.95,
                "ci_method": "two_sided_t_interval_across_matched_probe_fits",
                "f2llm_auc_mean": float(f_mean[j]),
                "f2llm_auc_sd": float(f_sd[j]),
                "f2llm_ci_lower": float(f_lo[j]),
                "f2llm_ci_upper": float(f_hi[j]),
                "ctclip_auc_mean": float(c_mean[j]),
                "ctclip_auc_sd": float(c_sd[j]),
                "ctclip_ci_lower": float(c_lo[j]),
                "ctclip_ci_upper": float(c_hi[j]),
                "paired_delta_mean": float(d_mean[j]),
                "paired_delta_sd": float(d_sd[j]),
                "paired_delta_ci_lower": float(d_lo[j]),
                "paired_delta_ci_upper": float(d_hi[j]),
            }
        )

    out = pd.DataFrame(rows)
    sector_rank = {name: i for i, name in enumerate(SECTOR_ORDER)}
    out["_sector_rank"] = out["sector"].map(sector_rank)
    out["_phecode_numeric"] = out["phecode"].astype(float)
    out = (
        out.sort_values(["_sector_rank", "_phecode_numeric"])
        .drop(columns=["_sector_rank", "_phecode_numeric"])
        .reset_index(drop=True)
    )

    status = {
        "status": "complete",
        "n_scans": int(len(y)),
        "n_phenotypes": len(labels),
        "n_matched_seeds": len(seeds),
        "seeds": seeds.tolist(),
        "ci_method": "two_sided_t_interval_across_matched_probe_fits",
        "ci_interpretation": "fit variation; not patient-sampling uncertainty",
        "t_critical_df19": t_critical,
        "inputs": {name: _relative(path) for name, path in paths.items()},
        "alignment_validation": {
            "seeds_equal": True,
            "labels_equal": True,
            "outcomes_equal": True,
            "h5_indices_equal": True,
            "labels_match_phecodes_csv": True,
            **macro_validation,
        },
        "macro_auc": {
            "f2llm_mean": float(seed_macro_f.mean()),
            "f2llm_sd": float(seed_macro_f.std(ddof=1)),
            "ctclip_mean": float(seed_macro_c.mean()),
            "ctclip_sd": float(seed_macro_c.std(ddof=1)),
            "paired_delta_mean": float((seed_macro_f - seed_macro_c).mean()),
            "paired_delta_sd": float((seed_macro_f - seed_macro_c).std(ddof=1)),
        },
    }
    return out, status


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
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
    return "".join(replacements.get(char, char) for char in text)


def _latex_breakable_group(value: Any) -> str:
    """Escape a group/sector label while permitting a portrait-column break at '/'."""
    return _latex_escape(value).replace("/", r"/\allowbreak{}")


def _longtable_header(
    caption: str,
    label: str,
    column_spec: str,
    headings: list[str],
    tabcolsep_pt: float = 2,
) -> list[str]:
    header = " & ".join(headings) + r" \\" 
    return [
        r"\begingroup",
        r"\normalsize",
        rf"\setlength{{\tabcolsep}}{{{tabcolsep_pt:g}pt}}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt plus 1fill}",
        rf"\begin{{longtable}}{{@{{}}{column_spec}@{{}}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{len(headings)}}}{{c}}{{\tablename\ \thetable\ -- continued}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        rf"\multicolumn{{{len(headings)}}}{{r}}{{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]


def write_zero_shot_latex(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "% Generated by experiments/supplementary/generate_auc_tables.py.",
        "% Portrait-only; requires: \\usepackage{array,booktabs,longtable}.",
        "",
    ]
    model_panels = [
        ("a", ["ours", "merlin", "ctclip"]),
        ("b", ["medsiglip", "biomedclip", "openai_clip"]),
    ]
    displayed_models = [model for _, models in model_panels for model in models]
    for dataset, spec in DATASETS.items():
        subset = df[df["dataset"].eq(dataset)]
        n_resamples = int(subset["n_boot"].max())
        for panel_index, (panel_tag, panel_models) in enumerate(model_panels, start=1):
            headings = [
                r"\textbf{Finding}",
                r"\textbf{$n_{+}$}",
            ] + [
                rf"\textbf{{{_latex_escape(MODEL_SPECS[m]['display'])}}}"
                for m in panel_models
            ]
            caption = (
                rf"\textbf{{Zero-shot finding annotation on {_latex_escape(spec.display)} "
                rf"(model panel {panel_index} of {len(model_panels)}).}} "
                rf"Values are AUROC (95\% percentile CI) from "
                rf"{n_resamples:,} "
                r"shared patient-cluster bootstrap resamples. Bold indicates the "
                r"largest point estimate across all model panels for each finding. "
                r"$n_{+}$ is the number of positive test scans."
            )
            if len(panel_models) == 3:
                column_spec = (
                    r">{\raggedright\arraybackslash}p{0.260\linewidth}"
                    r">{\centering\arraybackslash}p{0.060\linewidth}"
                    + r">{\centering\arraybackslash}p{0.210\linewidth}" * 3
                )
            else:
                column_spec = (
                    r">{\raggedright\arraybackslash}p{0.410\linewidth}"
                    r">{\centering\arraybackslash}p{0.070\linewidth}"
                    + r">{\centering\arraybackslash}p{0.240\linewidth}" * 2
                )
            lines.extend(
                _longtable_header(
                    caption,
                    f"tab:supp-zs-{dataset.replace('_', '-')}-{panel_tag}",
                    column_spec,
                    headings,
                    tabcolsep_pt=2,
                )
            )

            canonical = pd.read_csv(EXP1 / f"analysis/tables/per_label_auc__{dataset}.csv")
            for _, meta in canonical.iterrows():
                label_rows = subset[subset["label"].eq(meta["label"])]
                points = (
                    label_rows[label_rows["model"].isin(displayed_models)]
                    .set_index("model")["auc"]
                    .dropna()
                )
                best = float(points.max()) if len(points) else float("nan")
                cells = [
                    _latex_escape(meta["display"]),
                    str(int(label_rows.iloc[0]["full_cohort_n_positive"])),
                ]
                for model in panel_models:
                    row = label_rows[label_rows["model"].eq(model)].iloc[0]
                    point = row["auc"]
                    if pd.isna(point):
                        cell = r"\textemdash"
                    elif row["availability_status"] == "complete":
                        cell = rf"\mbox{{{point:.3f} ({row['ci_lower']:.3f}, {row['ci_upper']:.3f})}}"
                    else:
                        cell = (
                            rf"\shortstack{{{point:.3f}$^{{\dagger}}$\\"
                            r"CI unavailable}"
                        )
                    if pd.notna(point) and np.isclose(
                        float(point), best, atol=5e-13, rtol=0
                    ):
                        cell = rf"\textbf{{{cell}}}"
                    cells.append(cell)
                lines.append(" & ".join(cells) + " \\\\")
            lines.extend([r"\end{longtable}", r"\endgroup"])
    path.write_text("\n".join(lines) + "\n")


def write_linear_probe_latex(df: pd.DataFrame, path: Path) -> None:
    caption = (
        r"\textbf{Matched linear-probe performance for all 221 INSPECT phenotypes.} "
        r"Values are mean AUROC (two-sided 95\% $t$-based CI) across 20 "
        r"matched probe fits. Intervals quantify fit variation, not patient-sampling "
        r"uncertainty. $n_{+}$ gives the positive count among 2,612 test scans; "
        r"the remaining scans are negative. "
        r"Bold indicates the larger model point estimate per phenotype."
    )
    headings = [
        r"\textbf{Phecode}",
        r"\hspace*{0.5em}\textbf{Phenotype}",
        r"\textbf{$n_{+}$}",
        r"\textbf{ACT}",
        r"\textbf{CT-CLIP}",
    ]
    lines = [
        "% Generated by experiments/supplementary/generate_auc_tables.py.",
        "% Portrait-only; requires: \\usepackage{array,booktabs,longtable}.",
        "",
        *_longtable_header(
            caption,
            "tab:supp-linear-probe-221",
            r">{\centering\arraybackslash}p{0.095\linewidth}"
            r">{\raggedright\arraybackslash}p{0.355\linewidth}"
            r">{\centering\arraybackslash}p{0.075\linewidth}"
            r">{\centering\arraybackslash}p{0.220\linewidth}"
            r">{\centering\arraybackslash}p{0.220\linewidth}",
            headings,
            tabcolsep_pt=2,
        ),
    ]
    for _, row in df.iterrows():
        f_cell = (
            rf"\mbox{{{row['f2llm_auc_mean']:.3f} "
            rf"({row['f2llm_ci_lower']:.3f}, {row['f2llm_ci_upper']:.3f})}}"
        )
        c_cell = (
            rf"\mbox{{{row['ctclip_auc_mean']:.3f} "
            rf"({row['ctclip_ci_lower']:.3f}, {row['ctclip_ci_upper']:.3f})}}"
        )
        if row["f2llm_auc_mean"] >= row["ctclip_auc_mean"]:
            f_cell = rf"\textbf{{{f_cell}}}"
        else:
            c_cell = rf"\textbf{{{c_cell}}}"
        cells = [
            _latex_escape(row["phecode"]),
            r"\hspace*{0.5em}" + _latex_escape(row["phenotype"]),
            str(int(row["n_positive"])),
            f_cell,
            c_cell,
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([r"\end{longtable}", r"\endgroup"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.n_boot <= 0 or args.bootstrap_chunk <= 0:
        raise ValueError("--n-boot and --bootstrap-chunk must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    zero_shot, zero_status = generate_zero_shot(
        n_boot=args.n_boot,
        seed=args.seed,
        chunk_size=args.bootstrap_chunk,
        validation_tol=args.validation_tol,
    )
    linear_probe, linear_status = generate_linear_probe()

    paths = {
        "zero_shot_csv": output_dir / "zero_shot_auc_ci_tidy.csv",
        "zero_shot_latex": output_dir / "zero_shot_auc_ci_longtable.tex",
        "linear_probe_csv": output_dir / "linear_probe_221_auc_ci.csv",
        "linear_probe_latex": output_dir / "linear_probe_221_auc_ci_longtable.tex",
        "status_json": output_dir / "supplementary_auc_status.json",
    }
    zero_shot.to_csv(paths["zero_shot_csv"], index=False)
    linear_probe.to_csv(paths["linear_probe_csv"], index=False)
    write_zero_shot_latex(zero_shot, paths["zero_shot_latex"])
    write_linear_probe_latex(linear_probe, paths["linear_probe_latex"])

    status = {
        "schema_version": 1,
        "generator": _relative(Path(__file__)),
        "zero_shot": zero_status,
        "linear_probe_221": linear_status,
        "outputs": {name: _relative(path) for name, path in paths.items()},
    }
    paths["status_json"].write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    if zero_status["validation"]["n_discrepancies"]:
        raise RuntimeError(
            f"canonical AUROC validation failed for "
            f"{zero_status['validation']['n_discrepancies']} rows; see {paths['status_json']}"
        )

    print("\nGenerated supplementary artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(
        "Validation: max zero-shot |recomputed - canonical| = "
        f"{zero_status['validation']['max_absolute_difference']:.3g}; "
        "linear-probe archives aligned."
    )


if __name__ == "__main__":
    main()

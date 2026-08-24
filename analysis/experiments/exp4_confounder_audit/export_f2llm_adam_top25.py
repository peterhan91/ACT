#!/usr/bin/env python3
"""Export the exact Adam-20 positive top-25 concepts for all INSPECT labels.

This is a versioned extension of the full-bank top-10 audit used for the
Figure 5 coefficient convention.  It recomputes rankings directly from the
twenty Adam (lr=3e-2) probe bundles and all 376,194 f2llm concept embeddings;
it does not reuse the separate L-BFGS top-100 export and does not retrain a
probe.  The first ten ranks, AUROCs, and seed-level coefficients are checked
against ``f2llm_20seed_natural_top10_clinical_audit.json`` before output is
accepted.

The exported values are global probe--observation projections.  They do not
show that an observation occurred in an individual scan, caused a prediction,
or was used as a patient-level shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

DEFAULT_OUTPUT = RESULTS / "f2llm_adam20_top25_all221.json"
DEFAULT_EMBEDDINGS = ROOT / "_emb_f2llm_tmp.npy"
DEFAULT_BANK = ROOT / "concept_bank.f2llm_emb.npz"
DEFAULT_REFERENCE = RESULTS / "f2llm_20seed_natural_top10_clinical_audit.json"
SEED_TEMPLATE = (
    ROOT
    / "outputs/v1/external/"
    "phenotype__linear_f2llm__seed{seed}__lr3e-2__coefs.pt"
)

N_SEEDS = 20
TOP_K = 25
REFERENCE_K = 10
N_CONCEPTS_EXPECTED = 376_194
T_CRIT_DF19 = 2.093024054408263
COEFFICIENT_ATOL = 5e-5
AUC_ATOL = 1e-12


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def file_identity(path: Path, *, include_sha256: bool) -> dict[str, Any]:
    stat = path.stat()
    identity: dict[str, Any] = {
        "path": relative_or_absolute(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        identity["sha256"] = sha256(path)
    return identity


def load_seed_data() -> tuple[
    np.ndarray,
    list[str],
    list[str],
    dict[str, np.ndarray],
    list[Path],
]:
    """Return W[seed,label,dimension], labels, phecodes, AUROCs, and paths."""
    weights: list[np.ndarray] = []
    aucs: dict[str, list[float]] = {}
    labels_ref: list[str] | None = None
    phecodes_ref: list[str] | None = None
    paths: list[Path] = []

    for seed in range(1, N_SEEDS + 1):
        path = Path(str(SEED_TEMPLATE).format(seed=seed))
        if not path.exists():
            raise FileNotFoundError(f"missing seed bundle: {path}")
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        labels = [str(item) for item in bundle["labels"]]
        phecodes = [str(item) for item in bundle["phecodes"]]
        if labels_ref is None:
            labels_ref = labels
            phecodes_ref = phecodes
            aucs = {label: [] for label in labels}
        else:
            if labels != labels_ref:
                raise ValueError(f"label order mismatch: {path}")
            if phecodes != phecodes_ref:
                raise ValueError(f"phecode order mismatch: {path}")
        if int(bundle.get("seed", seed)) != seed:
            raise ValueError(f"seed metadata mismatch: {path}")
        if not math.isclose(
            float(bundle["lr"]), 0.03, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"expected Adam lr=3e-2: {path}")
        weight = bundle["W"].numpy().astype(np.float32, copy=False)
        if weight.ndim != 2 or weight.shape[0] != len(labels):
            raise ValueError(f"unexpected probe weight shape {weight.shape}: {path}")
        weights.append(weight)
        for label in labels:
            aucs[label].append(float(bundle["test_per_label_auc"][label]))
        paths.append(path)

    assert labels_ref is not None and phecodes_ref is not None
    stacked = np.stack(weights, axis=0)
    if stacked.shape[0] != N_SEEDS:
        raise ValueError(f"expected {N_SEEDS} seed bundles, got {stacked.shape[0]}")
    return (
        stacked,
        labels_ref,
        phecodes_ref,
        {label: np.asarray(values, dtype=np.float64) for label, values in aucs.items()},
        paths,
    )


def bank_mean(embeddings: np.memmap, chunk_size: int = 20_000) -> np.ndarray:
    """Match the canonical top-10 builder's full-bank mean calculation."""
    total = np.zeros(embeddings.shape[1], dtype=np.float64)
    for start in range(0, embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, embeddings.shape[0])
        total += embeddings[start:stop].astype(np.float64).sum(axis=0)
    return (total / embeddings.shape[0]).astype(np.float32)


def centered_normalized(rows: np.ndarray, mean: np.ndarray) -> np.ndarray:
    out = rows.astype(np.float32) - mean
    out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-9)
    return out


def full_bank_topk(
    weights: np.ndarray,
    embedding_path: Path,
    *,
    top_k: int = TOP_K,
    chunk_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Stream the exact raw-coefficient positive top-k for every phenotype."""
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] != N_CONCEPTS_EXPECTED:
        raise ValueError(f"unexpected f2llm embedding shape: {embeddings.shape}")
    if weights.shape[2] != embeddings.shape[1]:
        raise ValueError(
            f"probe/embedding dimension mismatch: {weights.shape} vs {embeddings.shape}"
        )
    if not 1 <= top_k <= embeddings.shape[0]:
        raise ValueError(f"invalid top-k {top_k}")

    print("computing the full-bank embedding mean", flush=True)
    mean = bank_mean(embeddings)
    mean_weight = weights.mean(axis=0, dtype=np.float32)
    n_labels = mean_weight.shape[0]
    best_scores = np.full((n_labels, top_k), -np.inf, dtype=np.float32)
    best_indices = np.full((n_labels, top_k), -1, dtype=np.int64)

    n_chunks = math.ceil(embeddings.shape[0] / chunk_size)
    for chunk_number, start in enumerate(
        range(0, embeddings.shape[0], chunk_size), start=1
    ):
        stop = min(start + chunk_size, embeddings.shape[0])
        concept_rows = centered_normalized(embeddings[start:stop], mean)
        scores = mean_weight @ concept_rows.T

        local_k = min(top_k, scores.shape[1])
        local_columns = np.argpartition(
            scores, scores.shape[1] - local_k, axis=1
        )[:, -local_k:]
        local_scores = np.take_along_axis(scores, local_columns, axis=1)
        local_indices = local_columns.astype(np.int64) + start

        merged_scores = np.concatenate([best_scores, local_scores], axis=1)
        merged_indices = np.concatenate([best_indices, local_indices], axis=1)
        keep_columns = np.argpartition(
            merged_scores, merged_scores.shape[1] - top_k, axis=1
        )[:, -top_k:]
        best_scores = np.take_along_axis(merged_scores, keep_columns, axis=1)
        best_indices = np.take_along_axis(merged_indices, keep_columns, axis=1)

        if chunk_number == 1 or chunk_number % 5 == 0 or chunk_number == n_chunks:
            print(
                f"ranked chunk {chunk_number}/{n_chunks} "
                f"({stop:,}/{embeddings.shape[0]:,} concepts)",
                flush=True,
            )

    for label_idx in range(n_labels):
        order = sorted(
            range(top_k),
            key=lambda position: (
                -float(best_scores[label_idx, position]),
                int(best_indices[label_idx, position]),
            ),
        )
        best_scores[label_idx] = best_scores[label_idx, order]
        best_indices[label_idx] = best_indices[label_idx, order]

    if np.any(best_indices < 0):
        raise RuntimeError("streamed full-bank result contains unfilled indices")
    if not np.all(best_scores[:, :-1] >= best_scores[:, 1:]):
        raise RuntimeError("streamed full-bank result is not descending")
    return best_indices, best_scores, mean, tuple(embeddings.shape)


def selected_embedding_matrix(
    embedding_path: Path,
    indices: np.ndarray,
    mean: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    embeddings = np.load(embedding_path, mmap_mode="r")
    unique = np.asarray(sorted({int(item) for item in indices.ravel()}), dtype=np.int64)
    rows = centered_normalized(embeddings[unique], mean)
    positions = {int(index): position for position, index in enumerate(unique)}
    return rows, positions


def auc_summary(values: np.ndarray) -> dict[str, Any]:
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half_width = T_CRIT_DF19 * std / math.sqrt(N_SEEDS)
    return {
        "mean": mean,
        "std_across_seeds": std,
        "ci95_mean": [mean - half_width, mean + half_width],
        "n_seeds": N_SEEDS,
        "ci_method": "two-sided t interval across 20 initialization seeds (df=19)",
        "per_seed": [float(item) for item in values],
    }


def assert_close(
    observed: float,
    expected: float,
    *,
    atol: float,
    context: str,
) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=atol):
        raise ValueError(
            f"{context}: observed {observed:.12g}, expected {expected:.12g}, "
            f"absolute tolerance {atol}"
        )


def validate_against_reference(
    phenotypes: list[dict[str, Any]],
    reference_path: Path,
) -> dict[str, Any]:
    """Require exact rank identity and numeric agreement for canonical ranks 1-10."""
    reference = json.loads(reference_path.read_text())
    reference_entries = reference.get("phenotypes", [])
    if len(reference_entries) != len(phenotypes):
        raise ValueError(
            f"reference phenotype count mismatch: {len(reference_entries)} vs "
            f"{len(phenotypes)}"
        )
    by_name = {entry["phenotype"]: entry for entry in reference_entries}
    if len(by_name) != len(reference_entries):
        raise ValueError("reference contains duplicate phenotype names")

    numeric_values_checked = 0
    ranked_rows_checked = 0
    for entry in phenotypes:
        label = entry["phenotype"]
        if label not in by_name:
            raise ValueError(f"phenotype absent from reference: {label}")
        expected = by_name[label]
        if str(entry["phecode"]) != str(expected["phecode"]):
            raise ValueError(f"phecode mismatch for {label}")

        for key in ("mean", "std_across_seeds"):
            assert_close(
                float(entry["auroc"][key]),
                float(expected["auroc"][key]),
                atol=AUC_ATOL,
                context=f"{label} AUROC {key}",
            )
            numeric_values_checked += 1
        for position, (observed_value, expected_value) in enumerate(
            zip(entry["auroc"]["per_seed"], expected["auroc"]["per_seed"]),
            start=1,
        ):
            assert_close(
                float(observed_value),
                float(expected_value),
                atol=AUC_ATOL,
                context=f"{label} AUROC seed {position}",
            )
            numeric_values_checked += 1

        observed_rows = entry["natural_top25"][:REFERENCE_K]
        expected_rows = expected["natural_top10"]
        if len(observed_rows) != REFERENCE_K or len(expected_rows) != REFERENCE_K:
            raise ValueError(f"malformed top-10 rows for {label}")
        for rank, (observed, expected_row) in enumerate(
            zip(observed_rows, expected_rows), start=1
        ):
            if int(observed["natural_rank"]) != rank:
                raise ValueError(f"malformed rank for {label}: {rank}")
            if int(observed["concept_index"]) != int(expected_row["concept_index"]):
                raise ValueError(f"concept-index mismatch for {label}, rank {rank}")
            if observed["concept"] != expected_row["concept"]:
                raise ValueError(f"concept-string mismatch for {label}, rank {rank}")
            for key in ("mean", "std"):
                assert_close(
                    float(observed["coefficient"][key]),
                    float(expected_row["coefficient"][key]),
                    atol=COEFFICIENT_ATOL,
                    context=f"{label}, rank {rank}, coefficient {key}",
                )
                numeric_values_checked += 1
            assert_close(
                float(observed["coefficient_of_mean_weight"]),
                float(expected_row["coefficient_of_mean_weight"]),
                atol=COEFFICIENT_ATOL,
                context=f"{label}, rank {rank}, mean-weight coefficient",
            )
            numeric_values_checked += 1
            for seed, (observed_value, expected_value) in enumerate(
                zip(
                    observed["coefficient"]["per_seed"],
                    expected_row["coefficient"]["per_seed"],
                ),
                start=1,
            ):
                assert_close(
                    float(observed_value),
                    float(expected_value),
                    atol=COEFFICIENT_ATOL,
                    context=f"{label}, rank {rank}, coefficient seed {seed}",
                )
                numeric_values_checked += 1
            ranked_rows_checked += 1

    expected_metadata = reference.get("metadata", {})
    if int(expected_metadata.get("concept_bank_size", -1)) != N_CONCEPTS_EXPECTED:
        raise ValueError("reference does not identify the expected full concept bank")
    return {
        "passed": True,
        "reference": relative_or_absolute(reference_path),
        "reference_sha256": sha256(reference_path),
        "phenotypes_checked": len(phenotypes),
        "ranked_rows_checked": ranked_rows_checked,
        "numeric_values_checked": numeric_values_checked,
        "rank_identity_required": True,
        "coefficient_absolute_tolerance": COEFFICIENT_ATOL,
        "auroc_absolute_tolerance": AUC_ATOL,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--concept-bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--reference-top10", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"refusing to overwrite versioned artifact: {args.output}; "
            "choose a new --output path"
        )
    if args.chunk_size < TOP_K:
        raise ValueError(f"--chunk-size must be at least {TOP_K}")

    weights, labels, phecodes, auc_by_label, seed_paths = load_seed_data()
    top_indices, top_scores, mean_vector, embedding_shape = full_bank_topk(
        weights,
        args.embeddings,
        top_k=TOP_K,
        chunk_size=args.chunk_size,
    )

    with np.load(args.concept_bank, allow_pickle=True) as concept_bank:
        concepts = concept_bank["concepts"]
        bank_size = int(len(concepts))
        if bank_size != N_CONCEPTS_EXPECTED or embedding_shape[0] != bank_size:
            raise ValueError("embedding/concept-bank row-count mismatch")
        concept_strings = [str(concepts[int(index)]) for index in top_indices.ravel()]
    concept_strings_array = np.asarray(concept_strings, dtype=object).reshape(
        top_indices.shape
    )

    selected_rows, row_positions = selected_embedding_matrix(
        args.embeddings, top_indices, mean_vector
    )
    phenotypes: list[dict[str, Any]] = []
    for label_idx, (label, phecode) in enumerate(zip(labels, phecodes)):
        indices = [int(item) for item in top_indices[label_idx]]
        positions = [row_positions[index] for index in indices]
        vectors = selected_rows[positions]
        per_seed_matrix = weights[:, label_idx, :] @ vectors.T
        rows: list[dict[str, Any]] = []
        for rank, (concept_index, selected_score) in enumerate(
            zip(indices, top_scores[label_idx]), start=1
        ):
            per_seed = per_seed_matrix[:, rank - 1]
            assert_close(
                float(per_seed.mean()),
                float(selected_score),
                atol=COEFFICIENT_ATOL,
                context=f"{label}, rank {rank}, recomputed coefficient",
            )
            rows.append(
                {
                    "natural_rank": rank,
                    "concept_index": concept_index,
                    "concept": str(concept_strings_array[label_idx, rank - 1]),
                    "coefficient": {
                        "mean": float(per_seed.mean()),
                        "std": float(per_seed.std(ddof=1)),
                        "per_seed": [float(item) for item in per_seed],
                    },
                    "coefficient_of_mean_weight": float(selected_score),
                }
            )
        phenotypes.append(
            {
                "phenotype": label,
                "phecode": phecode,
                "auroc": auc_summary(auc_by_label[label]),
                "natural_top25": rows,
            }
        )
        if (label_idx + 1) % 50 == 0 or label_idx + 1 == len(labels):
            print(
                f"assembled {label_idx + 1}/{len(labels)} phenotype records",
                flush=True,
            )

    validation = validate_against_reference(phenotypes, args.reference_top10)
    seed_identities = {
        path.name: file_identity(path, include_sha256=True) for path in seed_paths
    }
    output = {
        "metadata": {
            "schema_version": "f2llm-adam20-top25-v1",
            "generated_on": date.today().isoformat(),
            "task": "INSPECT-221 phenotype linear probe",
            "method": "f2llm",
            "optimizer": "Adam",
            "learning_rate": 0.03,
            "n_initialization_seeds": N_SEEDS,
            "top_k": TOP_K,
            "concept_bank_size": bank_size,
            "concept_embedding_dim": embedding_shape[1],
            "ranking": (
                "Positive top-25 recomputed over all 376,194 concepts as "
                "mean_seed(W[label]) dot normalize(raw_embedding - full_bank_mean); "
                "no deduplication, clinical filtering, replacement, or reranking."
            ),
            "coefficient": (
                "raw W[label] dot centered and L2-normalized f2llm concept embedding"
            ),
            "phenotype_order": "label order stored in the twenty matched seed bundles",
            "claim_boundary": (
                "Global probe-observation alignment only; not evidence that an observation "
                "occurred in an individual scan, caused a prediction, or was used as a "
                "patient-level shortcut."
            ),
            "source_files": {
                "seed_bundle_pattern": relative_or_absolute(SEED_TEMPLATE),
                "seed_bundles": seed_identities,
                "embeddings": file_identity(args.embeddings, include_sha256=False),
                "concept_bank": file_identity(args.concept_bank, include_sha256=False),
            },
            "provenance_checks": {
                "seed_bundle_count": len(seed_paths),
                "labels_in_identical_order_across_seeds": True,
                "phecodes_in_identical_order_across_seeds": True,
                "adam_learning_rate_matches_all_seeds": True,
                "embedding_bank_row_count_matches": True,
                "all_ranked_coefficients_match_per_seed_means": True,
                "bank_mean_l2_norm": float(np.linalg.norm(mean_vector)),
                "canonical_top10_validation": validation,
            },
        },
        "summary": {
            "n_phenotypes": len(phenotypes),
            "n_positive_ranked_pairs": len(phenotypes) * TOP_K,
            "n_phenotypes_with_mean_test_auroc_ge_0_75": sum(
                entry["auroc"]["mean"] >= 0.75 for entry in phenotypes
            ),
        },
        "phenotypes": phenotypes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}", flush=True)
    print(f"sha256 {sha256(args.output)}", flush=True)
    print(json.dumps(output["summary"], indent=2), flush=True)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()

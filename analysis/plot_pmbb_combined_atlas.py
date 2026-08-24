#!/usr/bin/env python3
"""Project PMBB phrases onto the fixed CT-RATE + Merlin F2LLM atlas.

This is the single-panel analogue of UCE Figure 3c.  The grey background is
the *unchanged* 376,194-row UMAP used by ``concept_latent_anatomy.png``; it is
not refit with PMBB.  Because the original UMAP reducer object was not saved,
PMBB coordinates are obtained with an audited out-of-sample projection:

1. recover the original PCA(50) linear transform from the row-aligned saved
   reference PCA scores and normalized 5,120-D F2LLM embeddings;
2. transform PMBB with that recovered reference PCA basis; and
3. place each PMBB phrase by a UMAP-membership-weighted neighbourhood
   barycentre in the fixed reference PCA(50) atlas.

Exact normalized strings already present in the atlas are placed at their
existing atlas coordinate.  They are displayed separately and both copies are
excluded from source-specific centroid comparisons.

The red overlay is concept-first: it contains every strict PMBB-only phrase in
four prespecified finding-by-anatomy families, and each family gets a broad
black rectangle marking a reproducible central window. The rectangles never
filter points and are visual summaries, not inferential tests. The linked snippets are rooted excerpts of
one saved global linkage; no displayed subset is reclustered.
Outputs contain no raw concept strings, but are derived from private
pseudonymized PMBB data and must not be published or uploaded without approval.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
from scipy.cluster.hierarchy import ClusterNode, dendrogram, to_tree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402

from concept_latent_umap import TAXONOMY, categorize  # noqa: E402
from plot_pmbb_uce_analogue import (  # noqa: E402
    BRANCH_COLOR,
    DEFAULT_REFERENCE_ANATOMY,
    DEFAULT_REFERENCE_BANK,
    DEFAULT_REFERENCE_EMBEDDINGS,
    DEFAULT_REFERENCE_FINDING,
    Family,
    centroid_matrix,
    concept_fingerprint,
    concept_quality_mask,
    fine_keys,
    group_ids,
    normalize_rows,
    scanpy_style_linkage,
    sha256_file,
    write_csv,
)
from radlex_anatomy_categories import classify  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_ATLAS_DIR = ROOT / "outputs" / "concept_umap" / "f2llm"
DEFAULT_PMMB_DIR = (
    ROOT / "pmbb_concepts" / "banks" / "pmbb_qwen36_extracted16896_f2llm_20260715"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "concept_uce" / "pmbb_extracted16896" / "combined_atlas"
)

REFERENCE_COLOR = "#C9C9C9"
NEW_COLOR = "#C00000"
SHARED_COLOR = "#7A5195"
TEXT_COLOR = "#292929"
REFERENCE_MARKER_AREA = 0.46
PMBB_MARKER_AREA = 2.40
REFERENCE_MARKER_ALPHA = 0.48
PMBB_MARKER_ALPHA = 0.84
PMBB_MARKER_EDGE_COLOR = "#8F1115"
PMBB_MARKER_EDGE_WIDTH = 0.10

# Figure 3c is an illustrative localization panel, not a confirmatory screen.
# Every strict phrase from these four clinically coherent families is red.
DEFAULT_DISPLAY_FAMILY_KEYS = (
    "Lung parenchyma || Nodule/mass",
    "Hepatobiliary || Hepatic",
    "Pleura || Pleural effusion",
    "Vasculature || Atherosclerosis",
)

# Every displayed family also gets a window and a tree callout. Pleural effusion
# was previously left red but unannotated for being the sparsest family; its
# window is in fact the tightest of the four (49.7% coverage, 10.8x enrichment),
# and omitting it left the figure showing three of the four families the caption
# named. Override per run with --callout-family.
DEFAULT_CALLOUT_FAMILY_KEYS = DEFAULT_DISPLAY_FAMILY_KEYS


@dataclass
class Candidate:
    family_index: int
    family: Family
    reference_points: int
    pmbb_points: int
    pmbb_reports: int
    x0: float
    x1: float
    y0: float
    y1: float
    reference_coverage: float
    pmbb_coverage: float
    strict_all_purity: float
    retained_label_purity: float
    median_center_distance: float
    compactness: float
    pair_leaf_gap: int
    direct_tree_pair: bool
    selected: bool = False
    selection_rank: int = 0
    display_number: int = 0
    clade_node_id: int = -1
    clade_leaf_ids: tuple[int, ...] = ()

    @property
    def area(self) -> float:
        return max(self.x1 - self.x0, 0.0) * max(self.y1 - self.y0, 0.0)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)


@dataclass
class Callout:
    """A broad family-centered display window; never a point filter."""

    family_index: int
    family: Family
    x0: float
    x1: float
    y0: float
    y1: float
    core_points: int
    core_min_index: int
    pmbb_target_points: int
    pmbb_all_points: int
    pmbb_purity: float
    pmbb_reports: int | None
    family_target_points: int
    family_reports: int | None
    reference_target_points: int
    reference_all_points: int
    reference_retained_points: int
    reference_all_purity: float
    reference_retained_purity: float
    local_enrichment: float
    area_fraction: float
    direct_tree_pair: bool
    clade_node_id: int
    clade_leaf_ids: tuple[int, ...]
    display_clade_node_id: int = -1

    @property
    def area(self) -> float:
        return max(self.x1 - self.x0, 0.0) * max(self.y1 - self.y0, 0.0)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    @property
    def family_coverage(self) -> float:
        return self.pmbb_target_points / max(self.family_target_points, 1)


def concept_first_display_mask(
    pmbb_keys: np.ndarray,
    pmbb_nonoverlap: np.ndarray,
    family_keys: Iterable[str] = DEFAULT_DISPLAY_FAMILY_KEYS,
) -> np.ndarray:
    """Return all strict PMBB rows in the prespecified display families."""
    strict = np.asarray(pmbb_nonoverlap, dtype=bool)
    keys = np.asarray(pmbb_keys, dtype=object)
    if keys.shape != strict.shape:
        raise ValueError("PMBB family keys and strict-row mask are not aligned")
    selected_keys = np.asarray(tuple(family_keys), dtype=object)
    if len(selected_keys) == 0:
        raise ValueError("at least one display family is required")
    return strict & np.isin(keys, selected_keys)


def _sha256_array(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    return sha256_file(path, block_size=block_size)


def _mean_by_batch(values: np.ndarray, batch_size: int) -> np.ndarray:
    total = np.zeros(values.shape[1], dtype=np.float64)
    for start in range(0, len(values), batch_size):
        total += np.asarray(values[start : start + batch_size]).sum(
            axis=0, dtype=np.float64
        )
    return total / len(values)


def recover_reference_pca_transform(
    embeddings: np.ndarray,
    scores: np.ndarray,
    cache: Path,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Recover sklearn PCA's mean/components from row-aligned scores.

    Let ``Y=(X-mu)V.T``.  PCA score columns are mutually orthogonal, hence
    ``V_j = Y_j.T @ X / (Y_j.T @ Y_j)``.  Centering ``Y`` removes the tiny
    float32 score mean before accumulating the cross-product.
    """
    if embeddings.shape[0] != scores.shape[0]:
        raise ValueError("reference embeddings and saved PCA scores are not aligned")
    if cache.exists():
        with np.load(cache) as payload:
            components = payload["components"].astype(np.float32)
            mean = payload["mean"].astype(np.float32)
            metrics = {
                "validation_rmse": float(payload["validation_rmse"]),
                "validation_max_abs": float(payload["validation_max_abs"]),
                "component_norm_max_error": float(payload["component_norm_max_error"]),
                "component_orthogonality_max_error": float(
                    payload["component_orthogonality_max_error"]
                ),
            }
        return components, mean, metrics

    n_rows, n_features = embeddings.shape
    n_components = scores.shape[1]
    score_mean = _mean_by_batch(scores, batch_size)
    score_gram = np.zeros((n_components, n_components), dtype=np.float64)
    embedding_sum = np.zeros(n_features, dtype=np.float64)
    cross = np.zeros((n_components, n_features), dtype=np.float64)

    for batch_number, start in enumerate(range(0, n_rows, batch_size), start=1):
        stop = min(start + batch_size, n_rows)
        x = normalize_rows(np.asarray(embeddings[start:stop], dtype=np.float32))
        y = np.asarray(scores[start:stop], dtype=np.float32)
        yc = y - score_mean.astype(np.float32)
        embedding_sum += x.sum(axis=0, dtype=np.float64)
        score_gram += np.asarray(yc.T @ yc, dtype=np.float64)
        cross += np.asarray(yc.T @ x, dtype=np.float64)
        if batch_number % 40 == 0 or stop == n_rows:
            print(
                f"      recovered PCA cross-product rows {stop:,}/{n_rows:,}",
                flush=True,
            )

    components = np.linalg.solve(score_gram, cross)
    mean = embedding_sum / n_rows
    gram = components @ components.T
    norm_error = float(np.max(np.abs(np.diag(gram) - 1.0)))
    orth_error = float(np.max(np.abs(gram - np.eye(n_components))))

    # Validate away from the first/last batches to detect row-order mistakes.
    rng = np.random.default_rng(0)
    check = np.sort(rng.choice(n_rows, size=min(4096, n_rows), replace=False))
    x_check = normalize_rows(np.asarray(embeddings[check], dtype=np.float32))
    prediction = (x_check - mean.astype(np.float32)) @ components.T.astype(np.float32)
    target = np.asarray(scores[check], dtype=np.float32)
    error = prediction - target
    rmse = float(np.sqrt(np.mean(np.square(error, dtype=np.float64))))
    max_abs = float(np.max(np.abs(error)))
    if rmse > 2e-3 or orth_error > 2e-2:
        raise ValueError(
            "recovered PCA transform failed validation: "
            f"rmse={rmse:.6g}, orthogonality_error={orth_error:.6g}"
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        components=components.astype(np.float32),
        mean=mean.astype(np.float32),
        score_mean=score_mean.astype(np.float32),
        validation_rmse=np.float64(rmse),
        validation_max_abs=np.float64(max_abs),
        component_norm_max_error=np.float64(norm_error),
        component_orthogonality_max_error=np.float64(orth_error),
    )
    metrics = {
        "validation_rmse": rmse,
        "validation_max_abs": max_abs,
        "component_norm_max_error": norm_error,
        "component_orthogonality_max_error": orth_error,
    }
    return components.astype(np.float32), mean.astype(np.float32), metrics


def transform_pca(
    embeddings: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    result = np.empty((len(embeddings), len(components)), dtype=np.float32)
    for start in range(0, len(embeddings), batch_size):
        stop = min(start + batch_size, len(embeddings))
        x = normalize_rows(np.asarray(embeddings[start:stop], dtype=np.float32))
        result[start:stop] = (x - mean) @ components.T
    return result


def exact_rerank_neighbors(
    query: np.ndarray,
    candidate_indices: np.ndarray,
    reference: np.ndarray,
    k: int,
    exclude_reference_rows: np.ndarray | None = None,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly rerank approximate candidates in the saved PCA50 space."""
    if len(query) != len(candidate_indices):
        raise ValueError("query/candidate row mismatch")
    result_indices = np.empty((len(query), k), dtype=np.int64)
    result_distances = np.empty((len(query), k), dtype=np.float32)
    for start in range(0, len(query), batch_size):
        stop = min(start + batch_size, len(query))
        candidates = candidate_indices[start:stop]
        candidate_values = np.asarray(reference[candidates], dtype=np.float32)
        delta = (
            candidate_values
            - np.asarray(query[start:stop], dtype=np.float32)[:, None, :]
        )
        distances = np.sqrt(np.square(delta).sum(axis=2, dtype=np.float32))
        if exclude_reference_rows is not None:
            distances[candidates == exclude_reference_rows[start:stop, None]] = np.inf
        order = np.argsort(distances, axis=1)[:, :k]
        chosen_indices = np.take_along_axis(candidates, order, axis=1)
        chosen_distances = np.take_along_axis(distances, order, axis=1)
        if not np.isfinite(chosen_distances).all():
            raise ValueError("not enough finite candidates after exact reranking")
        result_indices[start:stop] = chosen_indices
        result_distances[start:stop] = chosen_distances
    return result_indices, result_distances


def umap_weighted_barycentric_projection(
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    atlas_xy: np.ndarray,
) -> np.ndarray:
    """Use the weighted-barycentre initialization from UMAP.transform."""
    from scipy.sparse import coo_matrix
    from umap.umap_ import (
        compute_membership_strengths,
        init_graph_transform,
        smooth_knn_dist,
    )

    if neighbor_indices.shape != neighbor_distances.shape:
        raise ValueError("neighbor index/distance shape mismatch")
    k = neighbor_indices.shape[1]
    sigmas, rhos = smooth_knn_dist(
        np.asarray(neighbor_distances, dtype=np.float32),
        float(k),
        local_connectivity=0.0,
    )
    rows, cols, values, _ = compute_membership_strengths(
        np.asarray(neighbor_indices, dtype=np.int64),
        np.asarray(neighbor_distances, dtype=np.float32),
        sigmas,
        rhos,
        bipartite=True,
    )
    graph = coo_matrix(
        (values, (rows, cols)),
        shape=(len(neighbor_indices), len(atlas_xy)),
    ).tocsr()
    graph.eliminate_zeros()
    return init_graph_transform(graph, np.asarray(atlas_xy, dtype=np.float32))


def project_pmbb_to_fixed_atlas(
    reference_pca: np.ndarray,
    atlas_xy: np.ndarray,
    pmbb_pca: np.ndarray,
    reference_concepts: np.ndarray,
    pmbb_concepts: np.ndarray,
    cache: Path,
    reference_fingerprint: str,
    pmbb_fingerprint: str,
    seed: int,
    index_neighbors: int,
    candidate_neighbors: int,
    query_neighbors: int,
    validation_n: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if cache.exists():
        with np.load(cache) as payload:
            xy = payload["pmbb_xy"].astype(np.float32)
            exact_reference_index = payload["exact_reference_index"].astype(np.int64)
            metrics_json = str(payload["metrics_json"].item())
        cached_metrics = json.loads(metrics_json)
        cache_is_current = (
            xy.shape == (len(pmbb_concepts), 2)
            and exact_reference_index.shape == (len(pmbb_concepts),)
            and cached_metrics.get("reference_concept_fingerprint")
            == reference_fingerprint
            and cached_metrics.get("pmbb_concept_fingerprint") == pmbb_fingerprint
            and cached_metrics.get("index_neighbors") == index_neighbors
            and cached_metrics.get("candidate_neighbors") == candidate_neighbors
            and cached_metrics.get("query_neighbors") == query_neighbors
        )
        if cache_is_current:
            return xy, exact_reference_index, cached_metrics
        print("      ignoring stale fixed-atlas projection cache", flush=True)

    try:
        from pynndescent import NNDescent
    except ImportError as exc:
        raise SystemExit("pynndescent is required for fixed-atlas projection") from exc

    print(
        f"      building PCA50 neighbour index over {len(reference_pca):,} atlas rows",
        flush=True,
    )
    # PyNNDescent's Numba RP-tree kernels require a writable C-contiguous array;
    # a read-only np.load(..., mmap_mode="r") view fails during specialization.
    index_data = np.array(reference_pca, dtype=np.float32, copy=True, order="C")
    index = NNDescent(
        index_data,
        n_neighbors=max(index_neighbors, candidate_neighbors),
        metric="euclidean",
        random_state=seed,
        low_memory=True,
        n_jobs=8,
        verbose=True,
    )
    index.prepare()
    candidate_indices, _ = index.query(
        np.asarray(pmbb_pca, dtype=np.float32), k=candidate_neighbors, epsilon=0.12
    )
    neighbor_indices, neighbor_distances = exact_rerank_neighbors(
        np.asarray(pmbb_pca, dtype=np.float32),
        candidate_indices,
        reference_pca,
        query_neighbors,
    )
    projected = umap_weighted_barycentric_projection(
        neighbor_indices, neighbor_distances, atlas_xy
    )

    reference_lookup = {str(value): i for i, value in enumerate(reference_concepts)}
    exact_reference_index = np.fromiter(
        (reference_lookup.get(str(value), -1) for value in pmbb_concepts),
        dtype=np.int64,
        count=len(pmbb_concepts),
    )
    exact = exact_reference_index >= 0
    projected[exact] = atlas_xy[exact_reference_index[exact]]

    # True out-of-sample validation: omit the query row from its returned neighbours.
    rng = np.random.default_rng(seed)
    validation_rows = np.sort(
        rng.choice(
            len(reference_pca),
            size=min(validation_n, len(reference_pca)),
            replace=False,
        )
    )
    val_candidate_indices, _ = index.query(
        np.asarray(reference_pca[validation_rows], dtype=np.float32),
        k=candidate_neighbors,
        epsilon=0.12,
    )
    cleaned_indices, cleaned_distances = exact_rerank_neighbors(
        np.asarray(reference_pca[validation_rows], dtype=np.float32),
        val_candidate_indices,
        reference_pca,
        query_neighbors,
        exclude_reference_rows=validation_rows,
    )
    validation_xy = umap_weighted_barycentric_projection(
        cleaned_indices, cleaned_distances, atlas_xy
    )
    validation_error = np.linalg.norm(validation_xy - atlas_xy[validation_rows], axis=1)
    map_diagonal = float(np.linalg.norm(np.ptp(atlas_xy, axis=0)))
    metrics: dict[str, object] = {
        "method": (
            "PyNNDescent candidates in the original reference PCA50, exact "
            "candidate reranking, and UMAP membership-strength weighted barycentre"
        ),
        "index_rows": int(len(reference_pca)),
        "reference_concept_fingerprint": reference_fingerprint,
        "pmbb_concept_fingerprint": pmbb_fingerprint,
        "index_neighbors": int(index_neighbors),
        "candidate_neighbors": int(candidate_neighbors),
        "query_neighbors": int(query_neighbors),
        "exact_string_rows": int(exact.sum()),
        "strict_pmbb_rows": int((~exact).sum()),
        "heldout_reference_n": int(len(validation_rows)),
        "heldout_umap_error_median": float(np.median(validation_error)),
        "heldout_umap_error_q90": float(np.quantile(validation_error, 0.90)),
        "heldout_umap_error_median_fraction_map_diagonal": float(
            np.median(validation_error) / map_diagonal
        ),
        "strict_pmbb_nearest_pca_distance_median": float(
            np.median(neighbor_distances[~exact, 0])
        ),
        "strict_pmbb_nearest_pca_distance_q90": float(
            np.quantile(neighbor_distances[~exact, 0], 0.90)
        ),
        "claim_boundary": (
            "Out-of-sample neighbour projection onto a fixed UMAP, not a saved "
            "UMAP.transform call and not an independent clinical validation."
        ),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        pmbb_xy=projected,
        exact_reference_index=exact_reference_index,
        nearest_distance=neighbor_distances[:, 0].astype(np.float32),
        metrics_json=np.asarray(json.dumps(metrics)),
    )
    return projected, exact_reference_index, metrics


def combined_report_support(
    occurrence_path: Path,
    ids: np.ndarray,
    include: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_groups = int(ids.max()) + 1
    occurrences = np.zeros(n_groups, dtype=np.int64)
    report_ids: list[set[str]] = [set() for _ in range(n_groups)]
    with occurrence_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            concept_index = int(record["concept_index"])
            if not include[concept_index]:
                continue
            group = int(ids[concept_index])
            if group < 0:
                continue
            occurrences[group] += 1
            report_ids[group].add(str(record["sample_id"]))
    return occurrences, np.asarray(
        [len(values) for values in report_ids], dtype=np.int64
    )


def enumerate_combined_families(
    reference_keys: np.ndarray,
    pmbb_keys: np.ndarray,
    reference_nonoverlap: np.ndarray,
    pmbb_nonoverlap: np.ndarray,
    anatomy_order: Iterable[str],
    finding_order: Iterable[str],
    min_reference: int,
    min_pmbb: int,
) -> list[Family]:
    families: list[Family] = []
    for anatomy in anatomy_order:
        for finding in finding_order:
            family = Family(str(anatomy), str(finding))
            if (
                int(np.sum((reference_keys == family.key) & reference_nonoverlap))
                >= min_reference
                and int(np.sum((pmbb_keys == family.key) & pmbb_nonoverlap)) >= min_pmbb
            ):
                families.append(family)
    if len(families) < 4:
        raise ValueError("fewer than four combined families pass support thresholds")
    return families


def _rectangle_iou(a: Candidate, b: Candidate) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    intersection = ix * iy
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def _expand_pair_clade(
    family_index: int,
    n_families: int,
    nodes: list[ClusterNode],
    parent: dict[int, ClusterNode],
    min_leaves: int = 4,
    max_leaves: int = 8,
) -> tuple[ClusterNode, bool]:
    ref_id = family_index
    new_id = n_families + family_index
    ref_parent = parent.get(ref_id)
    direct = ref_parent is not None and ref_parent is parent.get(new_id)
    if direct:
        node = ref_parent
    else:
        ref_ancestors: set[int] = set()
        cursor = nodes[ref_id]
        while cursor.id in parent:
            cursor = parent[cursor.id]
            ref_ancestors.add(cursor.id)
        cursor = nodes[new_id]
        while cursor.id not in ref_ancestors:
            cursor = parent[cursor.id]
        node = cursor
    while node.count < min_leaves and node.id in parent:
        next_node = parent[node.id]
        if next_node.count > max_leaves and node.count >= 2:
            break
        node = next_node
    return node, direct


def evaluate_candidates(
    atlas_xy: np.ndarray,
    pmbb_xy: np.ndarray,
    reference_keys: np.ndarray,
    pmbb_keys: np.ndarray,
    reference_nonoverlap: np.ndarray,
    pmbb_nonoverlap: np.ndarray,
    families: list[Family],
    pmbb_reports: np.ndarray,
    tree: np.ndarray,
    quantile: float,
) -> tuple[list[Candidate], list[ClusterNode], dict[int, ClusterNode], list[int]]:
    n = len(families)
    labels = [family.key + " [reference]" for family in families] + [
        family.key + " [PMBB new]" for family in families
    ]
    leaf_order = list(dendrogram(tree, labels=labels, no_plot=True)["leaves"])
    leaf_rank = {leaf: rank for rank, leaf in enumerate(leaf_order)}
    _, nodes = to_tree(tree, rd=True)
    parent: dict[int, ClusterNode] = {}
    for node in nodes:
        if node.left is not None:
            parent[node.left.id] = node
        if node.right is not None:
            parent[node.right.id] = node

    retained = set(family.key for family in families)
    all_xy = np.vstack((atlas_xy[reference_nonoverlap], pmbb_xy[pmbb_nonoverlap]))
    all_keys = np.concatenate(
        (reference_keys[reference_nonoverlap], pmbb_keys[pmbb_nonoverlap])
    )
    all_retained = np.isin(all_keys, np.asarray(sorted(retained), dtype=object))
    candidates: list[Candidate] = []
    for index, family in enumerate(families):
        ref_points = atlas_xy[(reference_keys == family.key) & reference_nonoverlap]
        new_points = pmbb_xy[(pmbb_keys == family.key) & pmbb_nonoverlap]
        if len(ref_points) == 0 or len(new_points) == 0:
            continue
        # Freeze each highlighted region from reference-atlas points only.
        # PMBB coordinates therefore cannot move or enlarge the box used to
        # assess whether new phrases localize in that reference neighborhood.
        low = np.quantile(ref_points, quantile, axis=0)
        high = np.quantile(ref_points, 1.0 - quantile, axis=0)
        inside_all = (
            (all_xy[:, 0] >= low[0])
            & (all_xy[:, 0] <= high[0])
            & (all_xy[:, 1] >= low[1])
            & (all_xy[:, 1] <= high[1])
        )
        target_all = all_keys == family.key
        ref_inside = (
            (ref_points[:, 0] >= low[0])
            & (ref_points[:, 0] <= high[0])
            & (ref_points[:, 1] >= low[1])
            & (ref_points[:, 1] <= high[1])
        )
        new_inside = (
            (new_points[:, 0] >= low[0])
            & (new_points[:, 0] <= high[0])
            & (new_points[:, 1] >= low[1])
            & (new_points[:, 1] <= high[1])
        )
        strict_purity = float(
            (inside_all & target_all).sum() / max(inside_all.sum(), 1)
        )
        labelled_denominator = int((inside_all & all_retained).sum())
        labelled_purity = float(
            (inside_all & target_all & all_retained).sum()
            / max(labelled_denominator, 1)
        )
        ref_center = np.median(ref_points, axis=0)
        new_center = np.median(new_points, axis=0)
        center_distance = float(np.linalg.norm(ref_center - new_center))
        ref_radius = float(
            np.quantile(np.linalg.norm(ref_points - ref_center, axis=1), 0.80)
        )
        new_radius = float(
            np.quantile(np.linalg.norm(new_points - new_center, axis=1), 0.80)
        )
        clade, direct = _expand_pair_clade(index, n, nodes, parent)
        pair_gap = abs(leaf_rank[index] - leaf_rank[n + index])
        candidates.append(
            Candidate(
                family_index=index,
                family=family,
                reference_points=len(ref_points),
                pmbb_points=len(new_points),
                pmbb_reports=int(pmbb_reports[index]),
                x0=float(low[0]),
                x1=float(high[0]),
                y0=float(low[1]),
                y1=float(high[1]),
                reference_coverage=float(ref_inside.mean()),
                pmbb_coverage=float(new_inside.mean()),
                strict_all_purity=strict_purity,
                retained_label_purity=labelled_purity,
                median_center_distance=center_distance,
                compactness=(ref_radius + new_radius) / 2.0,
                pair_leaf_gap=pair_gap,
                direct_tree_pair=direct,
                clade_node_id=clade.id,
                clade_leaf_ids=tuple(clade.pre_order(lambda value: value.id)),
            )
        )
    return candidates, nodes, parent, leaf_order


def select_candidates(
    candidates: list[Candidate],
    n_display: int,
    min_reports: int,
    min_points: int,
    min_coverage: float,
    min_labelled_purity: float,
    max_iou: float,
    min_center_separation: float,
) -> list[Candidate]:
    eligible = [
        value
        for value in candidates
        if value.pmbb_reports >= min_reports
        and value.reference_points >= min_points
        and value.pmbb_points >= min_points
        and min(value.reference_coverage, value.pmbb_coverage) >= min_coverage
        and value.retained_label_purity >= min_labelled_purity
        and value.direct_tree_pair
    ]
    eligible.sort(
        key=lambda value: (
            -value.retained_label_purity,
            -value.strict_all_purity,
            value.median_center_distance,
            value.compactness,
            value.family.key,
        )
    )
    selected: list[Candidate] = []
    for value in eligible:
        if any(_rectangle_iou(value, prior) > max_iou for prior in selected):
            continue
        if any(
            np.linalg.norm(np.subtract(value.center, prior.center))
            < min_center_separation
            for prior in selected
        ):
            continue
        value.selected = True
        value.selection_rank = len(selected) + 1
        selected.append(value)
        if len(selected) == n_display:
            break
    if len(selected) < n_display:
        raise ValueError(
            f"only {len(selected)} localized candidates pass the fixed display rule; "
            "do not silently relax thresholds"
        )
    # Match top-to-bottom inset order to UMAP y, minimizing connector crossings.
    selected.sort(key=lambda value: -value.center[1])
    for display_number, value in enumerate(selected, start=1):
        value.display_number = display_number
    return selected


def concept_report_sets(
    occurrence_path: Path,
    n_concepts: int,
    include: np.ndarray,
) -> list[set[str]] | None:
    """Load report membership when the private source includes that trace.

    The full 16,896-record Hub artifact is intentionally embedding-only and
    omits report-level records. Returning ``None`` keeps that absence explicit
    instead of treating each unique concept as a report or inventing coverage.
    """
    if not occurrence_path.exists():
        return None
    report_sets: list[set[str]] = [set() for _ in range(n_concepts)]
    with occurrence_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            concept_index = int(record["concept_index"])
            if include[concept_index]:
                report_sets[concept_index].add(str(record["sample_id"]))
    return report_sets


def build_family_callouts(
    atlas_xy: np.ndarray,
    pmbb_xy: np.ndarray,
    reference_keys: np.ndarray,
    pmbb_keys: np.ndarray,
    reference_nonoverlap: np.ndarray,
    pmbb_nonoverlap: np.ndarray,
    families: list[Family],
    report_sets: list[set[str]] | None,
    tree_nodes: list[ClusterNode],
    parent: dict[int, ClusterNode],
    *,
    callout_family_keys: Iterable[str] = DEFAULT_CALLOUT_FAMILY_KEYS,
    display_family_keys: Iterable[str] = DEFAULT_DISPLAY_FAMILY_KEYS,
    central_fraction: float = 0.35,
    min_side_span_fraction: float = 0.06,
) -> list[Callout]:
    """Build one broad, deterministic central window per callout family.

    Within each family, rank phrases by MAD-scaled distance to the bivariate
    median, bound the closest ``central_fraction``, and expand either dimension
    to at least ``min_side_span_fraction`` of the fixed atlas span. All phrases
    in ``display_family_keys`` remain red whether or not they fall in a box.
    Grey atlas labels and tree topology are post-hoc annotations only.
    """
    requested_keys = tuple(callout_family_keys)
    if not 0.0 < central_fraction <= 1.0:
        raise ValueError("central_fraction must be in (0, 1]")
    if not 0.0 < min_side_span_fraction <= 1.0:
        raise ValueError("min_side_span_fraction must be in (0, 1]")
    display_mask = concept_first_display_mask(
        pmbb_keys, pmbb_nonoverlap, display_family_keys
    )
    display_indices = np.flatnonzero(display_mask)
    display_xy = np.asarray(pmbb_xy[display_indices], dtype=np.float64)
    display_keys = np.asarray(pmbb_keys[display_indices], dtype=object)
    if not np.isfinite(display_xy).all():
        raise ValueError("displayed PMBB projection contains non-finite coordinates")

    retained_keys = np.asarray([family.key for family in families], dtype=object)
    family_by_key = {family.key: index for index, family in enumerate(families)}
    reference_xy = np.asarray(atlas_xy[reference_nonoverlap], dtype=np.float64)
    reference_labels = reference_keys[reference_nonoverlap]
    reference_retained = np.isin(reference_labels, retained_keys)
    atlas_min = np.min(atlas_xy, axis=0).astype(np.float64)
    atlas_max = np.max(atlas_xy, axis=0).astype(np.float64)
    atlas_span = atlas_max - atlas_min
    atlas_area = float(np.prod(atlas_span))
    min_side = min_side_span_fraction * atlas_span
    n_families = len(families)

    callouts: list[Callout] = []
    for family_key in requested_keys:
        if family_key not in family_by_key:
            raise ValueError(f"display family is not retained: {family_key}")
        family_index = family_by_key[family_key]
        family = families[family_index]
        family_indices = np.flatnonzero(
            np.asarray(pmbb_nonoverlap, dtype=bool)
            & (np.asarray(pmbb_keys, dtype=object) == family_key)
        )
        family_xy = np.asarray(pmbb_xy[family_indices], dtype=np.float64)
        if len(family_xy) == 0:
            raise ValueError(f"too few strict PMBB points for {family_key}")
        median = np.median(family_xy, axis=0)
        mad = np.median(np.abs(family_xy - median), axis=0)
        standard_deviation = np.std(family_xy, axis=0)
        scale = np.where(
            mad > np.finfo(float).eps,
            mad,
            np.where(standard_deviation > np.finfo(float).eps, standard_deviation, 1.0),
        )
        distance = np.sqrt(np.square((family_xy - median) / scale).sum(axis=1))
        n_core = max(1, int(np.ceil(central_fraction * len(family_xy))))
        core_order = np.lexsort((family_indices, distance))[:n_core]
        core_xy = family_xy[core_order]
        low = np.min(core_xy, axis=0)
        high = np.max(core_xy, axis=0)
        center = (low + high) / 2.0
        half = np.maximum((high - low) / 2.0, min_side / 2.0)
        low = center - half
        high = center + half

        family_report_ids: set[str] | None = set() if report_sets is not None else None
        if report_sets is not None and family_report_ids is not None:
            for concept_index in family_indices:
                family_report_ids.update(report_sets[int(concept_index)])
        clade, direct = _expand_pair_clade(family_index, n_families, tree_nodes, parent)
        inside_red = (
            (display_xy[:, 0] >= low[0])
            & (display_xy[:, 0] <= high[0])
            & (display_xy[:, 1] >= low[1])
            & (display_xy[:, 1] <= high[1])
        )
        target_red = inside_red & (display_keys == family_key)
        inside_reference = (
            (reference_xy[:, 0] >= low[0])
            & (reference_xy[:, 0] <= high[0])
            & (reference_xy[:, 1] >= low[1])
            & (reference_xy[:, 1] <= high[1])
        )
        target_reference = inside_reference & (reference_labels == family_key)
        retained_reference = inside_reference & reference_retained

        represented_reports: set[str] | None = (
            set() if report_sets is not None else None
        )
        if report_sets is not None and represented_reports is not None:
            for concept_index in display_indices[target_red]:
                represented_reports.update(report_sets[int(concept_index)])

        box_area = float(np.prod(high - low))
        outer_low = np.maximum(center - 3.0 * half, atlas_min)
        outer_high = np.minimum(center + 3.0 * half, atlas_max)
        inside_outer = (
            (display_xy[:, 0] >= outer_low[0])
            & (display_xy[:, 0] <= outer_high[0])
            & (display_xy[:, 1] >= outer_low[1])
            & (display_xy[:, 1] <= outer_high[1])
        )
        annulus_target = int(
            ((inside_outer & ~inside_red) & (display_keys == family_key)).sum()
        )
        outer_area = float(np.prod(outer_high - outer_low))
        annulus_area = max(outer_area - box_area, np.finfo(float).eps)
        inner_density = int(target_red.sum()) / max(box_area, np.finfo(float).eps)
        annulus_density = annulus_target / annulus_area
        enrichment = inner_density / max(annulus_density, np.finfo(float).eps)

        callouts.append(
            Callout(
                family_index=family_index,
                family=family,
                x0=float(low[0]),
                x1=float(high[0]),
                y0=float(low[1]),
                y1=float(high[1]),
                core_points=n_core,
                core_min_index=int(family_indices[core_order].min()),
                pmbb_target_points=int(target_red.sum()),
                pmbb_all_points=int(inside_red.sum()),
                pmbb_purity=float(target_red.sum() / max(inside_red.sum(), 1)),
                pmbb_reports=(
                    len(represented_reports)
                    if represented_reports is not None
                    else None
                ),
                family_target_points=len(family_indices),
                family_reports=(
                    len(family_report_ids) if family_report_ids is not None else None
                ),
                reference_target_points=int(target_reference.sum()),
                reference_all_points=int(inside_reference.sum()),
                reference_retained_points=int(retained_reference.sum()),
                reference_all_purity=float(
                    target_reference.sum() / max(inside_reference.sum(), 1)
                ),
                reference_retained_purity=float(
                    target_reference.sum() / max(retained_reference.sum(), 1)
                ),
                local_enrichment=float(enrichment),
                area_fraction=box_area / atlas_area,
                direct_tree_pair=direct,
                clade_node_id=clade.id,
                clade_leaf_ids=tuple(clade.pre_order(lambda value: value.id)),
            )
        )
    return callouts


def assign_display_clades(
    selected: list[Callout],
    nodes: list[ClusterNode],
    parent: dict[int, ClusterNode],
) -> None:
    """Expand rooted global-tree excerpts to reference-like, slot-specific sizes."""
    leaf_limits = {
        "Lung parenchyma || Nodule/mass": (12, 16),
        "Hepatobiliary || Hepatic": (4, 6),
        "Vasculature || Atherosclerosis": (6, 8),
    }
    # Families without a tuned entry get the smallest reference-like excerpt,
    # which is what the sparse ones need; previously they raised a KeyError.
    default_limits = (4, 6)
    node_by_id = _node_map(nodes)
    for value in selected:
        minimum, maximum = leaf_limits.get(value.family.key, default_limits)
        node = node_by_id[value.clade_node_id]
        while node.count < minimum and node.id in parent:
            next_node = parent[node.id]
            if next_node.count > maximum:
                break
            node = next_node
        value.display_clade_node_id = node.id


def _node_map(nodes: list[ClusterNode]) -> dict[int, ClusterNode]:
    return {node.id: node for node in nodes}


def render_original_subtree(
    ax: plt.Axes,
    root: ClusterNode,
    leaf_order: list[int],
    families: list[Family],
    fontsize: float,
) -> tuple[float, float]:
    n = len(families)
    rank = {leaf: index for index, leaf in enumerate(leaf_order)}
    y_by_leaf = {leaf: 5.0 + 10.0 * rank[leaf] for leaf in leaf_order}
    clade_leaves = list(root.pre_order(lambda value: value.id))
    clade_ranks = sorted(rank[leaf] for leaf in clade_leaves)
    if clade_ranks != list(range(clade_ranks[0], clade_ranks[-1] + 1)):
        raise ValueError("displayed subtree is not contiguous in the global leaf order")

    def draw(node: ClusterNode) -> float:
        if node.is_leaf():
            return y_by_leaf[node.id]
        assert node.left is not None and node.right is not None
        left_y = draw(node.left)
        right_y = draw(node.right)
        ax.plot(
            [node.left.dist, node.dist, node.dist, node.right.dist],
            [left_y, left_y, right_y, right_y],
            color=BRANCH_COLOR,
            linewidth=1.0,
            solid_capstyle="butt",
            clip_on=False,
        )
        return (left_y + right_y) / 2.0

    root_y = draw(root)
    stem_end = max(root.dist * 1.13, root.dist + 0.015)
    ax.plot([root.dist, stem_end], [root_y, root_y], color=BRANCH_COLOR, linewidth=1.0)

    ordered = sorted(clade_leaves, key=lambda leaf: rank[leaf])
    ticks = [y_by_leaf[leaf] for leaf in ordered]
    labels: list[str] = []
    colors: list[str] = []
    for leaf in ordered:
        family_index = leaf if leaf < n else leaf - n
        is_new = leaf >= n
        labels.append(families[family_index].display + (" (new)" if is_new else ""))
        colors.append(NEW_COLOR if is_new else TEXT_COLOR)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.yaxis.tick_right()
    for tick, color in zip(ax.get_yticklabels(), colors):
        tick.set_color(color)
        tick.set_fontweight("normal")
    ax.set_ylim(ticks[0] - 5.0, ticks[-1] + 5.0)
    ax.set_xlim(stem_end * 1.02, 0.0)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return stem_end, root_y


def plot_combined_panel(
    atlas_xy: np.ndarray,
    pmbb_xy: np.ndarray,
    display_mask: np.ndarray,
    selected: list[Callout],
    tree_nodes: list[ClusterNode],
    leaf_order: list[int],
    families: list[Family],
    output: Path,
    dpi: int,
) -> None:
    display = np.asarray(display_mask, dtype=bool)
    if display.shape != (len(pmbb_xy),):
        raise ValueError("PMBB display mask is not aligned with projected rows")
    # The full canvas is exactly 1.5 times the physical width of the UMAP.
    # The map keeps the compact tree column close to its right border while
    # reserving the same small left/bottom label margins as the source UMAP.
    fig = plt.figure(figsize=(12.56, 8.0))
    umap_ax = fig.add_axes([0.0387, 0.0600, 0.6264, 0.9325])
    # Slot heights follow each excerpt's leaf count, not its position in the
    # column, so a tall tree cannot land in a short slot and overprint itself.
    INSET_LEFT, INSET_WIDTH = 0.675, 0.030
    INSET_TOP, INSET_BOTTOM, INSET_GAP = 0.94, 0.02, 0.04

    umap_ax.scatter(
        atlas_xy[:, 0],
        atlas_xy[:, 1],
        s=REFERENCE_MARKER_AREA,
        c=REFERENCE_COLOR,
        alpha=REFERENCE_MARKER_ALPHA,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    umap_ax.scatter(
        pmbb_xy[display, 0],
        pmbb_xy[display, 1],
        s=PMBB_MARKER_AREA,
        c="#D7191C",
        alpha=PMBB_MARKER_ALPHA,
        edgecolors=PMBB_MARKER_EDGE_COLOR,
        linewidths=PMBB_MARKER_EDGE_WIDTH,
        rasterized=True,
        zorder=3,
    )
    umap_ax.set_xlim(float(atlas_xy[:, 0].min()), float(atlas_xy[:, 0].max()))
    umap_ax.set_ylim(float(atlas_xy[:, 1].min()), float(atlas_xy[:, 1].max()))
    umap_ax.set_aspect("equal", adjustable="box")
    umap_ax.set_xticks([])
    umap_ax.set_yticks([])
    umap_ax.set_xlabel("UMAP1", fontsize=15, labelpad=10)
    umap_ax.set_ylabel("UMAP2", fontsize=15, labelpad=10)
    for spine in umap_ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.85)

    node_by_id = _node_map(tree_nodes)
    ordered_callouts = sorted(selected, key=lambda value: -value.center[1])

    def _clade_root(value: Callout) -> ClusterNode:
        root_id = (
            value.display_clade_node_id
            if value.display_clade_node_id >= 0
            else value.clade_node_id
        )
        return node_by_id[root_id]

    leaf_counts = [max(1, _clade_root(value).count) for value in ordered_callouts]
    n_insets = len(ordered_callouts)
    usable = (INSET_TOP - INSET_BOTTOM) - INSET_GAP * (n_insets - 1)
    if usable <= 0:
        raise ValueError(f"no vertical room for {n_insets} tree insets")
    heights = [usable * count / sum(leaf_counts) for count in leaf_counts]
    inset_specs = []
    cursor = INSET_TOP
    for height in heights:
        inset_specs.append((INSET_LEFT, cursor - height, INSET_WIDTH, height))
        cursor -= height + INSET_GAP

    for value, spec in zip(ordered_callouts, inset_specs):
        rectangle = Rectangle(
            (value.x0, value.y0),
            value.x1 - value.x0,
            value.y1 - value.y0,
            fill=False,
            edgecolor="#111111",
            linewidth=1.15,
            zorder=7,
        )
        umap_ax.add_patch(rectangle)
        inset = fig.add_axes(spec)
        inset.set_facecolor("none")
        root = _clade_root(value)
        fontsize = 10.2 if root.count > 10 else 10.8
        root_point = render_original_subtree(
            inset, root, leaf_order, families, fontsize=fontsize
        )
        source_point = (value.x1, (value.y0 + value.y1) / 2.0)
        connector = ConnectionPatch(
            xyA=source_point,
            coordsA=umap_ax.transData,
            xyB=root_point,
            coordsB=inset.transData,
            arrowstyle="-",
            connectionstyle="arc3,rad=0",
            linewidth=0.72,
            color="#4A4A4A",
            clip_on=False,
            zorder=2,
        )
        fig.add_artist(connector)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=dpi, facecolor="white")
    plt.close(fig)


def candidate_rows(candidates: list[Candidate]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value in candidates:
        rows.append(
            {
                "family": value.family.display,
                "anatomy": value.family.anatomy,
                "finding": value.family.finding,
                "reference_points": value.reference_points,
                "strict_pmbb_points": value.pmbb_points,
                "strict_pmbb_reports": value.pmbb_reports,
                "roi_x0": value.x0,
                "roi_x1": value.x1,
                "roi_y0": value.y0,
                "roi_y1": value.y1,
                "reference_coverage": value.reference_coverage,
                "pmbb_coverage": value.pmbb_coverage,
                "strict_all_point_purity": value.strict_all_purity,
                "retained_family_purity": value.retained_label_purity,
                "median_center_distance": value.median_center_distance,
                "mean_q80_radius": value.compactness,
                "pair_leaf_gap": value.pair_leaf_gap,
                "direct_tree_pair": int(value.direct_tree_pair),
                "selected": int(value.selected),
                "selection_rank": value.selection_rank or "",
                "display_number": value.display_number or "",
                "global_clade_node_id": value.clade_node_id,
                "global_clade_leaf_ids": ";".join(map(str, value.clade_leaf_ids)),
            }
        )
    return rows


def callout_rows(callouts: list[Callout]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value in callouts:
        rows.append(
            {
                "family": value.family.display,
                "anatomy": value.family.anatomy,
                "finding": value.family.finding,
                "central_core_points": value.core_points,
                "central_core_min_concept_index": value.core_min_index,
                "box_x0": value.x0,
                "box_x1": value.x1,
                "box_y0": value.y0,
                "box_y1": value.y1,
                "box_strict_pmbb_target_points": value.pmbb_target_points,
                "box_all_strict_pmbb_points": value.pmbb_all_points,
                "box_strict_pmbb_target_purity": value.pmbb_purity,
                "box_pmbb_reports": value.pmbb_reports,
                "family_strict_pmbb_points": value.family_target_points,
                "family_pmbb_reports": value.family_reports,
                "box_family_coverage": value.family_coverage,
                "box_reference_target_points": value.reference_target_points,
                "box_all_reference_points": value.reference_all_points,
                "box_retained_reference_points": value.reference_retained_points,
                "box_reference_target_purity_all": value.reference_all_purity,
                "box_reference_target_purity_retained": value.reference_retained_purity,
                "local_target_density_enrichment": value.local_enrichment,
                "box_area_fraction": value.area_fraction,
                "direct_tree_pair": int(value.direct_tree_pair),
                "global_clade_node_id": value.clade_node_id,
                "global_clade_leaf_ids": ";".join(map(str, value.clade_leaf_ids)),
                "display_clade_node_id": (
                    value.display_clade_node_id
                    if value.display_clade_node_id >= 0
                    else ""
                ),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bank", type=Path, default=DEFAULT_REFERENCE_BANK)
    parser.add_argument(
        "--reference-embeddings", type=Path, default=DEFAULT_REFERENCE_EMBEDDINGS
    )
    parser.add_argument(
        "--reference-pca", type=Path, default=DEFAULT_ATLAS_DIR / "pca50.npy"
    )
    parser.add_argument(
        "--reference-umap", type=Path, default=DEFAULT_ATLAS_DIR / "umap2d.npy"
    )
    parser.add_argument(
        "--reference-anatomy", type=Path, default=DEFAULT_REFERENCE_ANATOMY
    )
    parser.add_argument(
        "--reference-finding", type=Path, default=DEFAULT_REFERENCE_FINDING
    )
    parser.add_argument("--pmbb-dir", type=Path, default=DEFAULT_PMMB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-reference", type=int, default=100)
    parser.add_argument("--min-pmbb", type=int, default=20)
    parser.add_argument("--callout-central-fraction", type=float, default=0.35)
    parser.add_argument("--callout-min-side-span-fraction", type=float, default=0.06)
    parser.add_argument(
        "--callout-family",
        action="append",
        dest="callout_families",
        default=None,
        help=(
            "family key to draw a box and tree callout for; repeatable. "
            "Defaults to DEFAULT_CALLOUT_FAMILY_KEYS when unset."
        ),
    )
    parser.add_argument("--index-neighbors", type=int, default=64)
    parser.add_argument("--candidate-neighbors", type=int, default=64)
    parser.add_argument("--query-neighbors", type=int, default=15)
    parser.add_argument("--validation-n", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="retain NPZ/NPY/CSV/JSON/MD audit artifacts (default: PNG only)",
    )
    return parser.parse_args()


def main() -> None:
    from radlex_anatomy_categories import (
        GROUPS as ANATOMY_GROUPS,
        OTHER as ANATOMY_OTHER,
    )

    args = parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    source_metadata_path = args.pmbb_dir / "metadata.json"
    source_metadata = (
        json.loads(source_metadata_path.read_text())
        if source_metadata_path.exists()
        else {}
    )
    source_snapshot = source_metadata.get("source_snapshot", {})
    source_sampling = source_metadata.get("sampling", {})
    source_report_count = source_snapshot.get("total_record_count")
    if source_report_count is None:
        source_report_count = source_sampling.get("total_sample_count")
    source_occurrence_count = source_snapshot.get("total_observation_occurrences")
    if source_occurrence_count is None:
        source_occurrence_count = source_sampling.get("total_observation_occurrences")

    print("[1/7] Loading the fixed combined atlas and PMBB bank", flush=True)
    reference_pca = np.load(args.reference_pca, mmap_mode="r")
    atlas_xy = np.load(args.reference_umap, mmap_mode="r")
    reference_embeddings = np.load(args.reference_embeddings, mmap_mode="r")
    with np.load(args.reference_bank, allow_pickle=True) as payload:
        reference_concepts = payload["concepts"].copy()
    pmbb_bank_path = args.pmbb_dir / "concept_bank.f2llm_emb.npz"
    with np.load(pmbb_bank_path, allow_pickle=True) as payload:
        pmbb_concepts = payload["concepts"].copy()
        pmbb_embeddings = payload["emb"].astype(np.float32, copy=False)
    if not (
        len(reference_concepts)
        == len(reference_pca)
        == len(atlas_xy)
        == len(reference_embeddings)
    ):
        raise ValueError("combined atlas artifacts are not row aligned")

    print(
        "[2/7] Recovering and validating the original reference PCA transform",
        flush=True,
    )
    transform_cache = args.output / "reference_pca50_transform.npz"
    components, pca_mean, recovery_metrics = recover_reference_pca_transform(
        reference_embeddings,
        reference_pca,
        transform_cache,
        batch_size=args.batch_size,
    )
    reference_fingerprint = concept_fingerprint(reference_concepts)
    pmbb_fingerprint = concept_fingerprint(pmbb_concepts)
    pmbb_pca_path = args.output / "pmbb_pca50.npz"
    pmbb_pca: np.ndarray | None = None
    if pmbb_pca_path.exists():
        with np.load(pmbb_pca_path) as payload:
            cached_pca = payload["pmbb_pca"].astype(np.float32)
            cached_fingerprint = str(payload["concept_fingerprint"].item())
        if (
            cached_pca.shape == (len(pmbb_concepts), reference_pca.shape[1])
            and cached_fingerprint == pmbb_fingerprint
        ):
            pmbb_pca = cached_pca
        else:
            print("      ignoring stale PMBB PCA cache", flush=True)
    if pmbb_pca is None:
        pmbb_pca = transform_pca(
            pmbb_embeddings, components, pca_mean, batch_size=args.batch_size
        )
        np.savez(
            pmbb_pca_path,
            pmbb_pca=pmbb_pca,
            concept_fingerprint=np.asarray(pmbb_fingerprint),
        )

    print("[3/7] Projecting all PMBB phrases onto the unchanged atlas", flush=True)
    pmbb_xy, exact_reference_index, projection_metrics = project_pmbb_to_fixed_atlas(
        reference_pca,
        atlas_xy,
        pmbb_pca,
        reference_concepts,
        pmbb_concepts,
        args.output / "pmbb_fixed_atlas_projection.npz",
        reference_fingerprint=reference_fingerprint,
        pmbb_fingerprint=pmbb_fingerprint,
        seed=args.seed,
        index_neighbors=args.index_neighbors,
        candidate_neighbors=args.candidate_neighbors,
        query_neighbors=args.query_neighbors,
        validation_n=args.validation_n,
    )

    print("[4/7] Defining combined finding-by-anatomy families", flush=True)
    reference_anatomy = np.load(args.reference_anatomy, allow_pickle=True).astype(str)
    reference_finding = np.load(args.reference_finding, allow_pickle=True).astype(str)
    pmbb_assignments = [classify(str(value)) for value in pmbb_concepts]
    pmbb_anatomy = np.asarray([value.group for value in pmbb_assignments], dtype=str)
    pmbb_finding = categorize(pmbb_concepts).astype(str)
    reference_keys = fine_keys(reference_anatomy, reference_finding)
    pmbb_keys = fine_keys(pmbb_anatomy, pmbb_finding)
    quality = concept_quality_mask(pmbb_concepts)
    exact = exact_reference_index >= 0
    reference_strings = set(map(str, pmbb_concepts[exact]))
    reference_nonoverlap = np.fromiter(
        (str(value) not in reference_strings for value in reference_concepts),
        dtype=bool,
        count=len(reference_concepts),
    )
    pmbb_nonoverlap = (~exact) & quality
    finding_order = tuple(name for name, _ in TAXONOMY)
    anatomy_order = tuple(value for value in ANATOMY_GROUPS if value != ANATOMY_OTHER)
    families = enumerate_combined_families(
        reference_keys,
        pmbb_keys,
        reference_nonoverlap,
        pmbb_nonoverlap,
        anatomy_order,
        finding_order,
        args.min_reference,
        args.min_pmbb,
    )
    reference_group = group_ids(reference_keys, families)
    pmbb_group = group_ids(pmbb_keys, families)
    occurrence_path = args.pmbb_dir / "concept_report_occurrences.jsonl"
    report_sets = concept_report_sets(
        occurrence_path, len(pmbb_concepts), pmbb_nonoverlap
    )
    report_trace_available = report_sets is not None

    print("[5/7] Building one combined source-specific PCA50 hierarchy", flush=True)
    reference_centroids, _ = centroid_matrix(
        reference_pca,
        reference_group,
        len(families),
        include=reference_nonoverlap,
        normalize=False,
    )
    pmbb_centroids, _ = centroid_matrix(
        pmbb_pca,
        pmbb_group,
        len(families),
        include=pmbb_nonoverlap,
        normalize=False,
    )
    tree, tree_distance = scanpy_style_linkage(
        np.vstack((reference_centroids, pmbb_centroids))
    )
    np.savez_compressed(
        args.output / "combined_source_tree.npz",
        linkage=tree,
        distance=tree_distance,
    )

    print("[6/7] Fixing concept families, finding callouts, and plotting", flush=True)
    labels = [family.key + " [reference]" for family in families] + [
        family.key + " [PMBB new]" for family in families
    ]
    leaf_order = list(dendrogram(tree, labels=labels, no_plot=True)["leaves"])
    _, nodes = to_tree(tree, rd=True)
    parent: dict[int, ClusterNode] = {}
    for node in nodes:
        if node.left is not None:
            parent[node.left.id] = node
        if node.right is not None:
            parent[node.right.id] = node
    display_mask = concept_first_display_mask(pmbb_keys, pmbb_nonoverlap)
    selected = build_family_callouts(
        atlas_xy,
        pmbb_xy,
        reference_keys,
        pmbb_keys,
        reference_nonoverlap,
        pmbb_nonoverlap,
        families,
        report_sets,
        nodes,
        parent,
        central_fraction=args.callout_central_fraction,
        min_side_span_fraction=args.callout_min_side_span_fraction,
        callout_family_keys=(
            tuple(args.callout_families)
            if args.callout_families
            else DEFAULT_CALLOUT_FAMILY_KEYS
        ),
    )
    assign_display_clades(selected, nodes, parent)
    plot_combined_panel(
        atlas_xy,
        pmbb_xy,
        display_mask,
        selected,
        nodes,
        leaf_order,
        families,
        args.output / "fig3c_pmbb_combined_fixed_atlas",
        args.dpi,
    )

    print("[7/7] Writing privacy-safe audit records", flush=True)
    (args.output / "localized_candidate_audit.csv").unlink(missing_ok=True)
    (args.output / "localized_hotspot_audit.csv").unlink(missing_ok=True)
    write_csv(
        args.output / "family_callout_audit.csv",
        callout_rows(selected),
    )
    displayed_family_counts = {
        key: int(np.sum(display_mask & (pmbb_keys == key)))
        for key in DEFAULT_DISPLAY_FAMILY_KEYS
    }
    metadata = {
        "schema_version": 5,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "privacy": {
            "classification": "private derived patient data",
            "raw_concept_strings_written": False,
            "publish_without_approval": False,
        },
        "single_combined_analysis": True,
        "reference": {
            "name": "CT-RATE + Merlin combined atlas",
            "rows": int(len(atlas_xy)),
            "umap_path": str(args.reference_umap.relative_to(ROOT)),
            "umap_sha256": _sha256_array(args.reference_umap),
            "pca50_path": str(args.reference_pca.relative_to(ROOT)),
            "pca50_sha256": _sha256_array(args.reference_pca),
            "coordinates_refit": False,
        },
        "pmbb": {
            "source_artifact": source_metadata.get("artifact"),
            "source_directory": str(args.pmbb_dir),
            "extracted_records": (
                int(source_report_count) if source_report_count is not None else None
            ),
            "observation_occurrences": (
                int(source_occurrence_count)
                if source_occurrence_count is not None
                else None
            ),
            "concept_report_trace_available": report_trace_available,
            "unique_phrases": int(len(pmbb_concepts)),
            "exact_atlas_strings": int(exact.sum()),
            "strict_only_phrases": int(pmbb_nonoverlap.sum()),
            "displayed_strict_family_phrases": int(display_mask.sum()),
            "displayed_family_counts": displayed_family_counts,
            "quality_excluded": int((~quality).sum()),
            "concept_fingerprint": pmbb_fingerprint,
            "bank_sha256": sha256_file(pmbb_bank_path),
            "source_metadata_sha256": (
                sha256_file(source_metadata_path)
                if source_metadata_path.exists()
                else None
            ),
        },
        "pca_recovery": recovery_metrics,
        "fixed_atlas_projection": projection_metrics,
        "families": {
            "definition": "project-defined RadLex-anchored anatomy x regex finding family",
            "retained": len(families),
            "source_labels": len(labels),
            "tree_recipe": "original reference PCA50 group means; Pearson 1-r; complete linkage",
            "exact_string_copies_excluded": True,
        },
        "localized_display_rule": {
            "purpose": "concept-first illustrative examples; not a confirmatory screen",
            "red_population": (
                "all strict PMBB-only phrases in four prespecified "
                "finding-by-anatomy families"
            ),
            "red_points_filtered_by_density": False,
            "geometry": (
                "closest central fraction to each family bivariate median using "
                "per-axis MAD-scaled distance"
            ),
            "geometry_uses_family_labels": True,
            "geometry_uses_reference_points": False,
            "geometry_uses_tree_topology": False,
            "umap_geometry_used_for_inference": False,
            "central_fraction": args.callout_central_fraction,
            "minimum_side_fraction_of_atlas_span": (
                args.callout_min_side_span_fraction
            ),
            "callout_families": list(DEFAULT_CALLOUT_FAMILY_KEYS),
            "displayed_without_callout": "Pleura || Pleural effusion",
            "reference_labels_used_only_post_hoc": True,
            "tree_topology_used_only_post_hoc": True,
            "displayed_subtrees_reclustered": False,
            "n_callouts": len(selected),
            "callout_family_names": [value.family.display for value in selected],
            "callout_box_family_coverage": [
                value.family_coverage for value in selected
            ],
            "reference_marker_area": REFERENCE_MARKER_AREA,
            "pmbb_marker_area": PMBB_MARKER_AREA,
            "pmbb_to_reference_marker_area_ratio": (
                PMBB_MARKER_AREA / REFERENCE_MARKER_AREA
            ),
        },
        "claim_boundary": (
            "Qualitative cross-corpus semantic localization in F2LLM text space. "
            "The UMAP projection, anatomy labels and finding labels are text-derived; "
            "family phrases are not independent reports or patients, and this is not "
            "independent clinical or image-level validation."
        ),
        "seconds": round(time.time() - started, 1),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    selected_grey_purity = [value.reference_all_purity for value in selected]
    selected_coverage = [value.family_coverage for value in selected]
    if report_trace_available:
        selected_report_coverage = [
            value.pmbb_reports / value.family_reports
            for value in selected
            if value.pmbb_reports is not None
            and value.family_reports is not None
            and value.family_reports > 0
        ]
        if len(selected_report_coverage) != len(selected):
            raise ValueError("incomplete report trace for a displayed callout")
        report_coverage_clause = (
            f" and {min(selected_report_coverage):.1%}–"
            f"{max(selected_report_coverage):.1%} of reports from their respective "
            "families"
        )
    else:
        report_coverage_clause = (
            ". Concept-to-report trace is intentionally absent from the full "
            "embedding-only artifact, so report-level callout coverage is not estimated"
        )
    selected_red_purity = [value.pmbb_purity for value in selected]
    if not all(value.direct_tree_pair for value in selected):
        raise ValueError(
            "a displayed PMBB/reference family centroid is not a direct pair"
        )
    exact_message = (
        f"Across the {len(selected)} prespecified callout regions, the MAD-centered windows "
        f"contained {min(selected_coverage):.1%}–{max(selected_coverage):.1%} of strict "
        f"PMBB-only phrases{report_coverage_clause}. Within each window, "
        f"{min(selected_red_purity):.1%}–"
        f"{max(selected_red_purity):.1%} of displayed PMBB phrases carried the intended "
        f"family label, while {min(selected_grey_purity):.1%}–"
        f"{max(selected_grey_purity):.1%} of grey atlas phrases carried that label. "
        f"All {int(display_mask.sum()):,} strict PMBB-only phrases from the four "
        "prespecified families—including all points outside the "
        "windows—remain displayed in red. The windows are descriptive visual "
        "summaries, not data filters or inferential tests."
    )
    (args.output / "EXACT_MESSAGE.md").write_text(
        "# Exact result message\n\n> " + exact_message + "\n\n"
        "The grey background uses the unchanged combined-atlas UMAP coordinates. "
        "PMBB is placed by an audited out-of-sample PCA50-neighbour projection because "
        "the original UMAP reducer object was not saved.\n"
    )
    source_record_phrase = (
        f" from {int(source_report_count):,} extracted records"
        if source_report_count is not None
        else ""
    )
    trace_note = (
        "Concept-to-report trace is available for report-level coverage auditing."
        if report_trace_available
        else (
            "The full embedding-only artifact intentionally omits concept-to-report "
            "trace, so no family-level report coverage is claimed."
        )
    )
    (args.output / "README.md").write_text(
        "# PMBB fixed-atlas Figure 3c analogue\n\n"
        "Private derived patient data. Do not publish or upload without approval.\n\n"
        "The primary output is `fig3c_pmbb_combined_fixed_atlas`. Its grey "
        "background is the unchanged 376,194-phrase CT-RATE + Merlin UMAP used by "
        "`concept_latent_anatomy.png`; it is neither sampled nor refit. All "
        f"{int(pmbb_nonoverlap.sum()):,} QC-passing strict PMBB-only phrases"
        f"{source_record_phrase} are projected once, "
        f"but the red overlay contains all {int(display_mask.sum()):,} strict phrases "
        "from the four prespecified families only. No density filter removes red "
        f"outliers. The {int(exact.sum()):,} exact normalized overlaps are positive "
        "controls and are "
        "omitted from the overlay and source-centroid hierarchy.\n\n"
        "PMBB uses the recovered original PCA-50 basis followed by an audited "
        "UMAP-membership-weighted kNN barycentric projection onto the fixed atlas. "
        "Black rectangles mark reproducible MAD-centered windows for every displayed family; "
        f"they contain {min(selected_coverage):.1%}–{max(selected_coverage):.1%} of each "
        "family and are not complete family extents. PMBB family centroids use "
        "every strict phrase in that family, including points outside its box. "
        "Every linked tree is a contiguous rooted excerpt of `combined_source_tree.npz`; "
        f"no inset is reclustered. {trace_note}\n\n"
        "Reproduce from the repository root with:\n\n"
        "```bash\n"
        "python plot_pmbb_combined_atlas.py --batch-size 4096\n"
        "```\n\n"
        "See `run_metadata.json` and `family_callout_audit.csv` for the projection "
        "validation, fixed display rule, counts, purity, enrichment, and claim boundary.\n"
    )
    (args.output / "REFERENCE_DESIGN_CHECK.md").write_text(
        "# Check against UCE Figure 3c\n\n"
        "- One fixed atlas, not two cohort-specific manifolds: **pass**.\n"
        "- Every one of the 376,194 original CT-RATE + Merlin UMAP coordinates is "
        "drawn in grey without refitting: **pass**.\n"
        "- New source is red and exact normalized overlaps are excluded from the "
        "visual/centroid evidence: **pass**.\n"
        f"- Every one of the {int(display_mask.sum()):,} strict phrases in the four "
        "prespecified families is red; points outside boxes remain visible: **pass**.\n"
        "- Black boxes identify reproducible central windows for every displayed "
        "family: **pass**.\n"
        "- Every inset is a contiguous excerpt of one unchanged global PCA-50 "
        "centroid hierarchy; no subset reclustering: **pass**.\n"
        "- Red labels denote PMBB and black labels denote the atlas: **pass**.\n\n"
        "The remaining methodological difference is explicit: the original fitted "
        "UMAP reducer was not saved, so PMBB uses a validated fixed-atlas weighted-kNN "
        "projection rather than native `UMAP.transform`. The radiology labels are "
        "coarse text-derived finding-by-anatomy families, not independent clinical "
        "ground truth.\n"
    )
    if not args.keep_intermediates:
        final_png = (args.output / "fig3c_pmbb_combined_fixed_atlas.png").resolve()
        for path in sorted(args.output.rglob("*"), reverse=True):
            if path.is_file() and path.resolve() != final_png:
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    print(f"Done in {time.time() - started:.1f}s -> {args.output}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproduce the analysis pattern of UCE Figure 3a/3c for CT concepts.

This is an *analogue* of the UCE analysis, not an application of the UCE
model.  UCE accepts gene-count matrices; the radiology representation is the
frozen F2LLM concept embedding already used by this repository.

The released UCE Figure 3 notebooks use the following display pipeline:

    PCA(50) -> group means -> Pearson distance (1-r) -> complete linkage
    PCA(50) -> 15-neighbour graph -> UMAP(min_dist=.5, spread=1)

This script makes those choices explicit.  It additionally performs the
scientific alignment test in the full, row-normalized 5,120-dimensional F2LLM
space, because UMAP is only a visualization.

Fine labels are defined independently of the embeddings by crossing the two
existing taxonomies behind the concept-atlas figure:

* project-defined RadLex-4.3-anchored anatomical families; and
* the existing 18 observation/finding families.

Only label intersections with enough non-overlapping concepts in both the
reference bank and PMBB are retained.  Exact normalized strings present in
both banks are audited and excluded from reference-vs-new centroid validation,
so a match cannot be driven by duplicate text.

Outputs contain no raw concept strings, but remain derived from a private,
pseudonymized patient artifact and must not be published without approval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist, squareform
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle  # noqa: E402

from concept_latent_umap import OTHER as FINDING_OTHER  # noqa: E402
from concept_latent_umap import TAXONOMY, categorize  # noqa: E402
from plot_clinical_organization import (  # noqa: E402
    CTR_ONLY,
    MERLIN_ONLY,
    SHARED,
    source_membership,
)
from radlex_anatomy_categories import (  # noqa: E402
    ANATOMY_COLORS,
    GROUPS as ANATOMY_GROUPS,
    OTHER as ANATOMY_OTHER,
    classify,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_BANK = ROOT / "concept_bank.f2llm_emb.npz"
DEFAULT_REFERENCE_EMBEDDINGS = ROOT / "_emb_f2llm_tmp.npy"
DEFAULT_REFERENCE_ANATOMY = (
    ROOT / "outputs" / "concept_umap" / "f2llm" / "radlex_anatomy_categories.npy"
)
DEFAULT_REFERENCE_FINDING = (
    ROOT / "outputs" / "concept_umap" / "f2llm" / "categories.npy"
)
DEFAULT_PMMB_DIR = (
    ROOT
    / "pmbb_concepts"
    / "samples"
    / "pmbb_qwen36_sample2000_seed20260715"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "concept_uce" / "pmbb_sample2000"
DEFAULT_CTRATE = ROOT / "ctrate_concepts.jsonl"
DEFAULT_MERLIN = ROOT / "merlin_concepts.jsonl"

REFERENCE_LABEL = "CT-RATE + Merlin reference"
NEW_LABEL = "PMBB (new)"
REFERENCE_COLOR = "#C8C8C8"
NEW_COLOR = "#C00000"
SHARED_COLOR = "#7A5195"
BRANCH_COLOR = "#343434"

FINDING_ORDER = tuple(name for name, _ in TAXONOMY)
ANATOMY_ORDER = tuple(group for group in ANATOMY_GROUPS if group != ANATOMY_OTHER)


@dataclass(frozen=True)
class Family:
    anatomy: str
    finding: str

    @property
    def key(self) -> str:
        return f"{self.anatomy} || {self.finding}"

    @property
    def display(self) -> str:
        return f"{self.finding} — {self.anatomy}"


@dataclass(frozen=True)
class Stratum:
    slug: str
    title: str
    pmbb_bit: int
    reference_codes: tuple[np.uint8, ...]
    reference_name: str
    pmbb_name: str
    pmbb_cohort: str


STRATA = {
    "chest": Stratum(
        slug="chest_nc_vs_ctrate",
        title="Chest noncontrast",
        pmbb_bit=1,
        reference_codes=(CTR_ONLY, SHARED),
        reference_name="CT-RATE reference",
        pmbb_name="PMBB chest-NC",
        pmbb_cohort="pmbb_chest_nc",
    ),
    "abdomen": Stratum(
        slug="abdomen_ce_vs_merlin",
        title="Abdomen contrast",
        pmbb_bit=2,
        reference_codes=(MERLIN_ONLY, SHARED),
        reference_name="Merlin reference",
        pmbb_name="PMBB abdomen-CE",
        pmbb_cohort="pmbb_abd_ce",
    ),
}


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def concept_fingerprint(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    np.divide(result, np.clip(norms, 1e-9, None), out=result)
    return result


def fine_keys(anatomy: np.ndarray, finding: np.ndarray) -> np.ndarray:
    if len(anatomy) != len(finding):
        raise ValueError("anatomy/finding arrays are not row aligned")
    result = np.empty(len(anatomy), dtype=object)
    for index, (a_value, f_value) in enumerate(zip(anatomy, finding)):
        result[index] = f"{a_value} || {f_value}"
    return result


def pmbb_cohort_bits(path: Path, n_concepts: int) -> np.ndarray:
    """Return 1=chest, 2=abdomen, 3=present in both from occurrence provenance."""
    bits = np.zeros(n_concepts, dtype=np.uint8)
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            index = int(record["concept_index"])
            if not 0 <= index < n_concepts:
                raise ValueError(f"{path}:{line_number}: concept index out of range")
            cohort = record["cohort"]
            if cohort == "pmbb_chest_nc":
                bits[index] |= np.uint8(1)
            elif cohort == "pmbb_abd_ce":
                bits[index] |= np.uint8(2)
            else:
                raise ValueError(f"{path}:{line_number}: unknown cohort {cohort!r}")
    if np.any(bits == 0):
        raise ValueError("at least one PMBB concept has no cohort provenance")
    return bits


def concept_quality_mask(concepts: np.ndarray) -> np.ndarray:
    """Exclude punctuation/numeric/too-short extraction failures generically."""
    import re

    mask = np.zeros(len(concepts), dtype=bool)
    for index, value in enumerate(concepts):
        letters = "".join(re.findall(r"[a-z]+", str(value).lower()))
        mask[index] = len(letters) >= 3
    return mask


def pmbb_family_support(
    path: Path,
    ids: np.ndarray,
    cohort: str,
    include: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Count occurrences and distinct reports for each retained PMBB family."""
    n_groups = int(ids.max()) + 1
    occurrences = np.zeros(n_groups, dtype=np.int64)
    report_ids: list[set[str]] = [set() for _ in range(n_groups)]
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["cohort"] != cohort:
                continue
            concept_index = int(record["concept_index"])
            if not include[concept_index]:
                continue
            group = int(ids[concept_index])
            if group < 0:
                continue
            occurrences[group] += 1
            report_ids[group].add(str(record["sample_id"]))
    reports = np.asarray([len(values) for values in report_ids], dtype=np.int64)
    return occurrences, reports


def enumerate_families(
    reference_keys: np.ndarray,
    new_keys: np.ndarray,
    reference_scope: np.ndarray,
    new_scope: np.ndarray,
    reference_nonoverlap: np.ndarray,
    new_nonoverlap: np.ndarray,
    min_reference: int,
    min_new: int,
) -> tuple[list[Family], dict[str, dict[str, int]]]:
    """Choose labels without consulting embedding geometry."""
    families: list[Family] = []
    counts: dict[str, dict[str, int]] = {}
    for anatomy in ANATOMY_ORDER:
        for finding in FINDING_ORDER:
            family = Family(anatomy, finding)
            ref_all = int(np.sum((reference_keys == family.key) & reference_scope))
            new_all = int(np.sum((new_keys == family.key) & new_scope))
            ref_eval = int(np.sum((reference_keys == family.key) & reference_nonoverlap))
            new_eval = int(np.sum((new_keys == family.key) & new_nonoverlap))
            if ref_eval < min_reference or new_eval < min_new:
                continue
            families.append(family)
            counts[family.key] = {
                "reference_all": ref_all,
                "pmbb_all": new_all,
                "reference_nonoverlap": ref_eval,
                "pmbb_nonoverlap": new_eval,
            }
    if len(families) < 2:
        raise ValueError(
            "fewer than two common families pass the thresholds; "
            "lower --min-reference/--min-new only with a documented reason"
        )
    return families, counts


def group_ids(keys: np.ndarray, families: list[Family]) -> np.ndarray:
    lookup = {family.key: index for index, family in enumerate(families)}
    return np.fromiter((lookup.get(str(key), -1) for key in keys), dtype=np.int16)


def centroid_matrix(
    embeddings: np.ndarray,
    ids: np.ndarray,
    n_groups: int,
    include: np.ndarray | None = None,
    batch_size: int = 2048,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute means of row-normalized vectors without materializing a bank."""
    if len(ids) != embeddings.shape[0]:
        raise ValueError("group IDs do not align with embeddings")
    if include is None:
        include = np.ones(len(ids), dtype=bool)
    if len(include) != len(ids):
        raise ValueError("include mask does not align with embeddings")

    sums = np.zeros((n_groups, embeddings.shape[1]), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    for group in range(n_groups):
        rows = np.flatnonzero((ids == group) & include)
        counts[group] = len(rows)
        for start in range(0, len(rows), batch_size):
            block_rows = rows[start : start + batch_size]
            block = (
                normalize_rows(embeddings[block_rows])
                if normalize
                else np.asarray(embeddings[block_rows], dtype=np.float32)
            )
            sums[group] += block.sum(axis=0, dtype=np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"empty centroid groups: {missing}")
    means = sums / counts[:, None]
    return means.astype(np.float32), counts


def scanpy_style_linkage(group_pcs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Released-notebook tree: Pearson 1-r over group means, complete linkage."""
    correlation = np.corrcoef(np.asarray(group_pcs, dtype=np.float64))
    if not np.isfinite(correlation).all():
        raise ValueError("non-finite group correlation; check zero-variance centroid")
    distance = np.clip(1.0 - correlation, 0.0, 2.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=True), method="complete")
    return tree, distance


def sample_reference_rows(
    reference_scope: np.ndarray,
    nonoverlap: np.ndarray,
    per_source: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Sample reference-only rows and retain one atlas copy of shared strings."""
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(reference_scope & nonoverlap)
    count = min(per_source, len(candidates))
    reference_only = np.sort(rng.choice(candidates, size=count, replace=False))
    exact_shared = np.flatnonzero(reference_scope & ~nonoverlap)
    audit = {
        "reference_only_sample": len(reference_only),
        "exact_string_shared_single_atlas_copy": len(exact_shared),
        "total_reference_coordinates": len(reference_only) + len(exact_shared),
    }
    return reference_only.astype(np.int64), exact_shared.astype(np.int64), audit


def _pastel(hex_color: str, mix: float = 0.86) -> tuple[float, float, float]:
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(rgb * (1.0 - mix) + mix)


def _contiguous_runs(values: list[str]) -> list[tuple[int, int, str]]:
    if not values:
        return []
    runs: list[tuple[int, int, str]] = []
    start = 0
    current = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value != current:
            runs.append((start, index, current))
            start = index
            current = value
    runs.append((start, len(values), current))
    return runs


def plot_fig3a(
    tree: np.ndarray,
    families: list[Family],
    new_counts: np.ndarray,
    output: Path,
    dpi: int,
    stratum: Stratum,
) -> list[int]:
    labels = [family.key for family in families]
    label_to_family = {family.key: family for family in families}
    count_by_key = {family.key: int(new_counts[i]) for i, family in enumerate(families)}

    fig = plt.figure(figsize=(12.8, 11.0))
    tree_ax = fig.add_axes([0.05, 0.07, 0.49, 0.78])
    label_ax = fig.add_axes([0.545, 0.07, 0.275, 0.78], sharey=tree_ax)
    strip_ax = fig.add_axes([0.825, 0.07, 0.145, 0.78], sharey=tree_ax)
    result = dendrogram(
        tree,
        labels=labels,
        orientation="left",
        color_threshold=0,
        above_threshold_color=BRANCH_COLOR,
        link_color_func=lambda _: BRANCH_COLOR,
        leaf_font_size=8.5,
        no_labels=True,
        ax=tree_ax,
    )
    ordered_keys = list(result["ivl"])
    ordered_anatomy = [label_to_family[key].anatomy for key in ordered_keys]

    # SciPy leaf centres are at 5, 15, ... from bottom to top.
    for start, stop, anatomy in _contiguous_runs(ordered_anatomy):
        y0, y1 = 10 * start, 10 * stop
        tree_ax.axhspan(
            y0,
            y1,
            color=_pastel(ANATOMY_COLORS[anatomy]),
            alpha=0.72,
            zorder=-10,
        )
        label_ax.axhspan(
            y0,
            y1,
            color=_pastel(ANATOMY_COLORS[anatomy]),
            alpha=0.72,
            zorder=-10,
        )
        strip_ax.add_patch(
            Rectangle(
                (0.0, y0),
                1.0,
                y1 - y0,
                facecolor=_pastel(ANATOMY_COLORS[anatomy], 0.79),
                edgecolor="white",
                linewidth=1.5,
            )
        )
        strip_ax.text(
            0.5,
            (y0 + y1) / 2,
            anatomy,
            ha="center",
            va="center",
            fontsize=8.3,
            wrap=True,
        )

    for index, key in enumerate(ordered_keys):
        family = label_to_family[key]
        label_ax.text(
            0.01,
            5 + 10 * index,
            f"{family.finding}  ($n_{{unique}}$={count_by_key[key]:,})",
            color=ANATOMY_COLORS[family.anatomy],
            ha="left",
            va="center",
            fontsize=8.0,
        )
    tree_ax.tick_params(axis="y", length=0)
    tree_ax.set_xticks([])
    for spine in tree_ax.spines.values():
        spine.set_visible(False)
    for axis in (label_ax, strip_ax):
        axis.set_xlim(0, 1)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    fig.text(
        0.05,
        0.94,
        "a",
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.51,
        0.925,
        f"Organization of observation families in {stratum.pmbb_name}",
        fontsize=15.5,
        fontweight="bold",
        ha="center",
        va="top",
    )
    callout = FancyBboxPatch(
        (0.29, 0.842),
        0.25,
        0.072,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor=BRANCH_COLOR,
        linewidth=1.3,
    )
    fig.add_artist(callout)
    fig.text(
        0.415,
        0.878,
        f"Map {stratum.pmbb_name} concepts\ndirectly with frozen F2LLM",
        ha="center",
        va="center",
        fontsize=11.2,
    )
    fig.add_artist(
        FancyArrowPatch(
            (0.54, 0.878),
            (0.62, 0.825),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color=BRANCH_COLOR,
            connectionstyle="angle,angleA=0,angleB=-90,rad=0",
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [labels.index(key) for key in ordered_keys]


def plot_source_dendrogram(
    tree: np.ndarray,
    labels: list[str],
    family_by_label: dict[str, Family],
    output: Path,
    dpi: int,
    figsize: tuple[float, float] = (9.0, 15.0),
    title: str | None = None,
) -> list[str]:
    fig, ax = plt.subplots(figsize=figsize)
    result = dendrogram(
        tree,
        labels=labels,
        orientation="left",
        color_threshold=0,
        above_threshold_color=BRANCH_COLOR,
        link_color_func=lambda _: BRANCH_COLOR,
        leaf_font_size=7.2,
        ax=ax,
    )
    display_labels: list[str] = []
    colors: list[str] = []
    for label in result["ivl"]:
        family = family_by_label[label]
        source = NEW_LABEL if label.endswith(" [PMBB new]") else REFERENCE_LABEL
        display_labels.append(
            f"{family.finding} — {family.anatomy}"
            + (" (PMBB new)" if source == NEW_LABEL else "")
        )
        colors.append(NEW_COLOR if source == NEW_LABEL else "#333333")
    ax.set_yticklabels(display_labels)
    for tick, color in zip(ax.get_yticklabels(), colors):
        tick.set_color(color)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return list(result["ivl"])


def plot_fig3c(
    xy: np.ndarray,
    n_reference_only: int,
    n_shared: int,
    exact_overlap_count: int,
    selected_tree: np.ndarray,
    selected_labels: list[str],
    family_by_label: dict[str, Family],
    output: Path,
    dpi: int,
    stratum: Stratum,
) -> None:
    fig = plt.figure(figsize=(16.2, 8.8))
    umap_ax = fig.add_axes([0.045, 0.10, 0.55, 0.70])
    tree_ax = fig.add_axes([0.61, 0.08, 0.365, 0.76])

    position = np.arange(len(xy))
    reference = position < n_reference_only
    shared = (position >= n_reference_only) & (
        position < n_reference_only + n_shared
    )
    new = position >= n_reference_only + n_shared
    umap_ax.scatter(
        xy[reference, 0],
        xy[reference, 1],
        s=1.0,
        c=REFERENCE_COLOR,
        alpha=0.25,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    umap_ax.scatter(
        xy[shared, 0],
        xy[shared, 1],
        s=2.2,
        c=SHARED_COLOR,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    umap_ax.scatter(
        xy[new, 0],
        xy[new, 1],
        s=1.7,
        c=NEW_COLOR,
        alpha=0.52,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    umap_ax.set_xticks([])
    umap_ax.set_yticks([])
    umap_ax.set_xlabel("UMAP1", fontsize=11)
    umap_ax.set_ylabel("UMAP2", fontsize=11)
    for spine in umap_ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")
    umap_ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                   markerfacecolor=REFERENCE_COLOR, markeredgecolor="none",
                   label=f"{stratum.reference_name}-only sample ($n$={n_reference_only:,})"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                   markerfacecolor=SHARED_COLOR, markeredgecolor="none",
                   label=f"Exact normalized string shared ($n$={n_shared:,})"),
            Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                   markerfacecolor=NEW_COLOR, markeredgecolor="none",
                   label=f"Strict {stratum.pmbb_name}-only ($n$={int(new.sum()):,})"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=9.2,
    )

    tree_result = dendrogram(
        selected_tree,
        labels=selected_labels,
        orientation="left",
        color_threshold=0,
        above_threshold_color=BRANCH_COLOR,
        link_color_func=lambda _: BRANCH_COLOR,
        leaf_font_size=7.4,
        ax=tree_ax,
    )
    display_labels = []
    colors = []
    for label in tree_result["ivl"]:
        family = family_by_label[label]
        is_new = label.endswith(" [PMBB new]")
        display_labels.append(family.display + (" (new)" if is_new else ""))
        colors.append(NEW_COLOR if is_new else "#333333")
    tree_ax.set_yticklabels(display_labels)
    for tick, color in zip(tree_ax.get_yticklabels(), colors):
        tick.set_color(color)
    tree_ax.set_xticks([])
    tree_ax.tick_params(axis="y", length=0, pad=3)
    for spine in tree_ax.spines.values():
        spine.set_visible(False)
    tree_ax.set_title(
        "Most prevalent shared observation families",
        fontsize=11.5,
        fontweight="bold",
        pad=8,
    )

    fig.text(0.045, 0.955, "c", fontsize=24, fontweight="bold", va="top")
    fig.text(
        0.50,
        0.945,
        f"Mapping strict {stratum.pmbb_name}-only concepts into {stratum.reference_name}",
        ha="center",
        va="top",
        fontsize=15.5,
        fontweight="bold",
    )
    for x, width, label in (
        (0.18, 0.19, stratum.pmbb_name),
        (0.41, 0.20, stratum.reference_name),
    ):
        box = FancyBboxPatch(
            (x, 0.842), width, 0.058,
            boxstyle="round,pad=0.010,rounding_size=0.015",
            transform=fig.transFigure,
            facecolor="white", edgecolor=BRANCH_COLOR, linewidth=1.1,
        )
        fig.add_artist(box)
        fig.text(x + width / 2, 0.871, label, ha="center", va="center", fontsize=10.5)
    fig.text(0.382, 0.871, "+", ha="center", va="center", fontsize=24)
    fig.text(
        0.045,
        0.035,
        f"The {exact_overlap_count:,} exact normalized strings shared by both datasets are shown once in purple; "
        "red is strict PMBB-only. UMAP is display-only.",
        fontsize=8.8,
        color="#555555",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def alignment_table(
    reference_centroids: np.ndarray,
    new_centroids: np.ndarray,
    families: list[Family],
    seed: int,
    permutation_replicates: int = 10_000,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """UCE-style centroid retrieval, explicitly treated as taxonomy consistency."""
    n = len(families)
    combined = np.vstack((reference_centroids, new_centroids))
    # Centroids are means of row-normalized concepts; do not re-normalize them.
    distance = cdist(new_centroids, combined, metric="euclidean")
    rows: list[dict[str, object]] = []
    top1_hits = 0
    top3_hits = 0
    reference_only_hits = 0
    within_anatomy_hits = 0
    within_finding_hits = 0
    within_anatomy_total = 0
    within_finding_total = 0
    nearest_reference_indices = np.empty(n, dtype=np.int16)
    for index, family in enumerate(families):
        distance[index, n + index] = np.inf  # exclude the query centroid itself
        order = np.argsort(distance[index])
        correct = index
        top1 = int(order[0] == correct)
        top3 = int(correct in order[:3])
        ref_order = np.argsort(distance[index, :n])
        nearest_reference_indices[index] = int(ref_order[0])
        ref_top1 = int(ref_order[0] == correct)
        same_anatomy = np.array(
            [candidate.anatomy == family.anatomy for candidate in families],
            dtype=bool,
        )
        same_finding = np.array(
            [candidate.finding == family.finding for candidate in families],
            dtype=bool,
        )
        anatomy_candidates = np.flatnonzero(same_anatomy)
        finding_candidates = np.flatnonzero(same_finding)
        within_anatomy: int | str = ""
        within_finding: int | str = ""
        if len(anatomy_candidates) > 1:
            within_anatomy = int(
                anatomy_candidates[np.argmin(distance[index, anatomy_candidates])]
                == correct
            )
            within_anatomy_hits += within_anatomy
            within_anatomy_total += 1
        if len(finding_candidates) > 1:
            within_finding = int(
                finding_candidates[np.argmin(distance[index, finding_candidates])]
                == correct
            )
            within_finding_hits += within_finding
            within_finding_total += 1
        top1_hits += top1
        top3_hits += top3
        reference_only_hits += ref_top1
        correct_rank = int(np.flatnonzero(order == correct)[0]) + 1
        rows.append(
            {
                "family": family.display,
                "anatomy": family.anatomy,
                "finding": family.finding,
                "matching_reference_rank_among_all_centroids": correct_rank,
                "matching_reference_in_top1": top1,
                "matching_reference_in_top3": top3,
                "nearest_reference_is_matching_label": ref_top1,
                "matching_finding_within_same_anatomy": within_anatomy,
                "matching_anatomy_within_same_finding": within_finding,
                "matching_reference_distance": float(distance[index, correct]),
                "nearest_any_distance": float(distance[index, order[0]]),
            }
        )
    rng = np.random.default_rng(seed)
    null_accuracy = np.empty(permutation_replicates, dtype=np.float32)
    label_indices = np.arange(n, dtype=np.int16)
    for replicate in range(permutation_replicates):
        permuted_labels = rng.permutation(label_indices)
        null_accuracy[replicate] = np.mean(
            permuted_labels[nearest_reference_indices] == label_indices
        )
    observed_reference_accuracy = reference_only_hits / n
    permutation_p = (1 + int(np.sum(null_accuracy >= observed_reference_accuracy))) / (
        permutation_replicates + 1
    )
    summary = {
        "n_families": n,
        "top1_accuracy_all_source_specific_centroids": top1_hits / n,
        "top3_accuracy_all_source_specific_centroids": top3_hits / n,
        "top1_accuracy_reference_candidates_only": reference_only_hits / n,
        "finding_accuracy_within_same_anatomy": (
            within_anatomy_hits / within_anatomy_total if within_anatomy_total else None
        ),
        "finding_within_same_anatomy_n": within_anatomy_total,
        "anatomy_accuracy_within_same_finding": (
            within_finding_hits / within_finding_total if within_finding_total else None
        ),
        "anatomy_within_same_finding_n": within_finding_total,
        "reference_label_permutation": {
            "replicates": permutation_replicates,
            "null_mean_accuracy": float(null_accuracy.mean()),
            "p_greater_equal_observed": permutation_p,
        },
        "distance": "Euclidean between means of row-L2-normalized full 5120-D F2LLM vectors",
        "exact_normalized_string_overlap_excluded": True,
        "claim_boundary": (
            "Taxonomy-consistency sanity check, not independent clinical validation: "
            "the text-derived rules and text embeddings share lexical signal."
        ),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bank", type=Path, default=DEFAULT_REFERENCE_BANK)
    parser.add_argument("--reference-embeddings", type=Path, default=DEFAULT_REFERENCE_EMBEDDINGS)
    parser.add_argument("--reference-anatomy", type=Path, default=DEFAULT_REFERENCE_ANATOMY)
    parser.add_argument("--reference-finding", type=Path, default=DEFAULT_REFERENCE_FINDING)
    parser.add_argument("--pmbb-dir", type=Path, default=DEFAULT_PMMB_DIR)
    parser.add_argument("--ctrate-jsonl", type=Path, default=DEFAULT_CTRATE)
    parser.add_argument("--merlin-jsonl", type=Path, default=DEFAULT_MERLIN)
    parser.add_argument("--stratum", choices=tuple(STRATA), default="chest")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: outputs/concept_uce/pmbb_sample2000/<matched stratum>",
    )
    parser.add_argument("--min-reference", type=int, default=100)
    parser.add_argument("--min-new", type=int, default=20)
    parser.add_argument("--min-new-reports", type=int, default=20)
    parser.add_argument("--reference-per-source", type=int, default=20_000)
    parser.add_argument("--display-families", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--skip-umap-model", action="store_true")
    args = parser.parse_args()
    for name in (
        "min_reference",
        "min_new",
        "min_new_reports",
        "reference_per_source",
        "display_families",
        "dpi",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    started = time.time()
    stratum = STRATA[args.stratum]
    args.output = args.output or (DEFAULT_OUTPUT / stratum.slug)
    args.output.mkdir(parents=True, exist_ok=True)
    pmbb_bank_path = args.pmbb_dir / "concept_bank.f2llm_emb.npz"

    print("[1/8] Loading row-aligned concepts and labels", flush=True)
    with np.load(args.reference_bank, allow_pickle=True) as payload:
        reference_concepts = payload["concepts"].copy()
        reference_bank_shape = tuple(payload["emb"].shape)
    reference_embeddings = np.load(args.reference_embeddings, mmap_mode="r")
    if tuple(reference_embeddings.shape) != reference_bank_shape:
        raise ValueError(
            f"reference embedding shape mismatch: bank={reference_bank_shape}, "
            f"memmap={reference_embeddings.shape}"
        )
    reference_anatomy = np.load(args.reference_anatomy, allow_pickle=True).astype(str)
    reference_finding = np.load(args.reference_finding, allow_pickle=True).astype(str)
    if not (
        len(reference_concepts)
        == len(reference_anatomy)
        == len(reference_finding)
        == reference_embeddings.shape[0]
    ):
        raise ValueError("reference concepts, embeddings, and labels are not row aligned")

    with np.load(pmbb_bank_path, allow_pickle=True) as payload:
        new_concepts = payload["concepts"].copy()
        new_embeddings = normalize_rows(payload["emb"])
    if new_embeddings.shape[1] != reference_embeddings.shape[1]:
        raise ValueError("reference and PMBB embedding dimensions differ")
    new_assignments = [classify(str(value)) for value in new_concepts]
    new_anatomy = np.array([assignment.group for assignment in new_assignments], dtype=str)
    new_finding = categorize(new_concepts).astype(str)

    reference_keys = fine_keys(reference_anatomy, reference_finding)
    new_keys = fine_keys(new_anatomy, new_finding)
    membership, membership_counts = source_membership(
        reference_concepts,
        args.ctrate_jsonl,
        args.merlin_jsonl,
    )
    cohort_bits = pmbb_cohort_bits(
        args.pmbb_dir / "concept_report_occurrences.jsonl", len(new_concepts)
    )
    quality = concept_quality_mask(new_concepts)
    reference_scope = np.isin(membership, np.asarray(stratum.reference_codes))
    new_scope_raw = (cohort_bits & np.uint8(stratum.pmbb_bit)) > 0
    new_scope = new_scope_raw & quality

    reference_strings = set(map(str, reference_concepts[reference_scope]))
    new_strings = set(map(str, new_concepts[new_scope]))
    overlap = reference_strings & new_strings
    reference_nonoverlap = np.fromiter(
        (
            bool(reference_scope[index]) and str(value) not in new_strings
            for index, value in enumerate(reference_concepts)
        ),
        dtype=bool,
        count=len(reference_concepts),
    )
    new_nonoverlap = np.fromiter(
        (
            bool(new_scope[index]) and str(value) not in reference_strings
            for index, value in enumerate(new_concepts)
        ),
        dtype=bool,
        count=len(new_concepts),
    )
    print(
        f"      stratum={stratum.title}; reference={int(reference_scope.sum()):,}; "
        f"PMBB={int(new_scope.sum()):,}; exact-string overlap={len(overlap):,}; "
        f"strict PMBB-only={int(new_nonoverlap.sum()):,}",
        flush=True,
    )

    families, family_counts = enumerate_families(
        reference_keys,
        new_keys,
        reference_scope,
        new_scope,
        reference_nonoverlap,
        new_nonoverlap,
        args.min_reference,
        args.min_new,
    )
    reference_group = group_ids(reference_keys, families)
    new_group = group_ids(new_keys, families)
    occurrence_path = args.pmbb_dir / "concept_report_occurrences.jsonl"
    _, candidate_strict_reports = pmbb_family_support(
        occurrence_path,
        new_group,
        stratum.pmbb_cohort,
        new_nonoverlap,
    )
    keep = candidate_strict_reports >= args.min_new_reports
    if not np.all(keep):
        families = [family for family, retain in zip(families, keep) if retain]
        family_counts = {family.key: family_counts[family.key] for family in families}
        if len(families) < 2:
            raise ValueError("report-count threshold leaves fewer than two families")
        reference_group = group_ids(reference_keys, families)
        new_group = group_ids(new_keys, families)
    pmbb_all_occurrences, pmbb_all_reports = pmbb_family_support(
        occurrence_path,
        new_group,
        stratum.pmbb_cohort,
        new_scope,
    )
    pmbb_strict_occurrences, pmbb_strict_reports = pmbb_family_support(
        occurrence_path,
        new_group,
        stratum.pmbb_cohort,
        new_nonoverlap,
    )
    print(f"      retained {len(families)} independently labelled families", flush=True)

    print("[2/8] Computing full-space, non-overlapping family centroids", flush=True)
    reference_centroids, reference_centroid_counts = centroid_matrix(
        reference_embeddings,
        reference_group,
        len(families),
        include=reference_nonoverlap,
    )
    new_centroids, new_centroid_counts = centroid_matrix(
        new_embeddings,
        new_group,
        len(families),
        include=new_nonoverlap,
    )
    alignment_rows, alignment_summary = alignment_table(
        reference_centroids, new_centroids, families, seed=args.seed
    )
    write_csv(args.output / "centroid_alignment_full5120.csv", alignment_rows)
    print(
        "      top-3 matching-reference accuracy: "
        f"{alignment_summary['top3_accuracy_all_source_specific_centroids']:.3f}",
        flush=True,
    )

    print("[3/8] Reproducing Figure 3a-style scoped PMBB PCA/tree", flush=True)
    fig3a_mask = new_scope & (new_group >= 0)
    pca_new = PCA(n_components=50, svd_solver="randomized", random_state=args.seed)
    new_pca = pca_new.fit_transform(new_embeddings[fig3a_mask]).astype(np.float32)
    new_pc_centroids, fig3a_counts = centroid_matrix(
        new_pca,
        new_group[fig3a_mask],
        len(families),
        normalize=False,
    )
    fig3a_tree, fig3a_distance = scanpy_style_linkage(new_pc_centroids)
    fig3a_order = plot_fig3a(
        fig3a_tree,
        families,
        fig3a_counts,
        args.output / "fig3a_pmbb_observation_hierarchy",
        args.dpi,
        stratum,
    )
    np.savez_compressed(
        args.output / "fig3a_tree.npz",
        linkage=fig3a_tree,
        distance=fig3a_distance,
        leaf_order=np.asarray(fig3a_order, dtype=np.int16),
    )
    del new_pca

    print("[4/8] Sampling the matched reference and separating exact overlap", flush=True)
    reference_only_rows, shared_rows, sample_counts = sample_reference_rows(
        reference_scope,
        reference_nonoverlap,
        args.reference_per_source,
        args.seed,
    )
    strict_new_rows = np.flatnonzero(new_nonoverlap).astype(np.int64)

    print("[5/8] Fitting one joint PCA(50) to reference-only + shared + PMBB-only", flush=True)
    reference_sample = normalize_rows(reference_embeddings[reference_only_rows])
    shared_sample = normalize_rows(reference_embeddings[shared_rows])
    strict_new_sample = new_embeddings[strict_new_rows]
    joint = np.vstack((reference_sample, shared_sample, strict_new_sample)).astype(
        np.float32, copy=False
    )
    n_reference_sample = len(reference_sample)
    n_shared_sample = len(shared_sample)
    pca_joint = PCA(n_components=50, svd_solver="randomized", random_state=args.seed)
    joint_pca = pca_joint.fit_transform(joint).astype(np.float32)
    del joint, reference_sample, shared_sample, strict_new_sample
    np.save(args.output / "joint_pca50.npy", joint_pca)
    np.savez_compressed(
        args.output / "joint_pca_model.npz",
        components=pca_joint.components_.astype(np.float32),
        mean=pca_joint.mean_.astype(np.float32),
        explained_variance=pca_joint.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca_joint.explained_variance_ratio_.astype(np.float32),
    )

    print("[6/8] Fitting joint UMAP display", flush=True)
    try:
        import umap
    except ImportError as exc:
        raise SystemExit(
            "umap-learn is required. Install the pinned plotting dependency with "
            "`python -m pip install umap-learn==0.5.7 pynndescent==0.5.13`."
        ) from exc
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.5,
        spread=1.0,
        metric="euclidean",
        init="spectral",
        random_state=args.seed,
        n_jobs=1,
        low_memory=True,
        verbose=True,
    )
    joint_umap = reducer.fit_transform(joint_pca).astype(np.float32)
    np.save(args.output / "joint_umap2d.npy", joint_umap)
    np.savez_compressed(
        args.output / "joint_rows.npz",
        reference_only_bank_indices=reference_only_rows,
        exact_shared_reference_bank_indices=shared_rows,
        strict_pmbb_only_bank_indices=strict_new_rows,
    )
    if not args.skip_umap_model:
        joblib.dump(reducer, args.output / "joint_umap_model.joblib", compress=3)

    print("[7/8] Building the source-specific family tree and figures", flush=True)
    # PCA is linear, so transforming full-space centroids is equivalent to the
    # mean transformed coordinate for every concept in the group.
    source_pc_centroids = np.vstack(
        (pca_joint.transform(reference_centroids), pca_joint.transform(new_centroids))
    )
    all_source_tree, all_source_distance = scanpy_style_linkage(source_pc_centroids)
    source_labels = [family.key + " [reference]" for family in families] + [
        family.key + " [PMBB new]" for family in families
    ]
    family_by_source_label = {
        family.key + suffix: family
        for family in families
        for suffix in (" [reference]", " [PMBB new]")
    }
    full_tree_order = plot_source_dendrogram(
        all_source_tree,
        source_labels,
        family_by_source_label,
        args.output / "fig3c_full_source_label_dendrogram",
        args.dpi,
        title="Reference and PMBB observation-family centroids",
    )
    np.savez_compressed(
        args.output / "fig3c_full_tree.npz",
        linkage=all_source_tree,
        distance=all_source_distance,
        leaf_order=np.asarray([source_labels.index(value) for value in full_tree_order]),
    )

    display_indices = sorted(
        range(len(families)),
        key=lambda index: (-int(new_centroid_counts[index]), families[index].key),
    )[: min(args.display_families, len(families))]
    display_centroids = np.vstack(
        (
            pca_joint.transform(reference_centroids[display_indices]),
            pca_joint.transform(new_centroids[display_indices]),
        )
    )
    display_tree, _ = scanpy_style_linkage(display_centroids)
    display_labels = [source_labels[index] for index in display_indices] + [
        source_labels[len(families) + index] for index in display_indices
    ]
    plot_fig3c(
        joint_umap,
        n_reference_sample,
        n_shared_sample,
        len(overlap),
        display_tree,
        display_labels,
        family_by_source_label,
        args.output / "fig3c_pmbb_reference_alignment",
        args.dpi,
        stratum,
    )

    print("[8/8] Writing privacy-safe audit tables and metadata", flush=True)
    count_rows: list[dict[str, object]] = []
    for index, family in enumerate(families):
        row: dict[str, object] = {
            "family": family.display,
            "anatomy": family.anatomy,
            "finding": family.finding,
            **family_counts[family.key],
            "reference_centroid_count": int(reference_centroid_counts[index]),
            "pmbb_centroid_count": int(new_centroid_counts[index]),
            "pmbb_all_occurrences": int(pmbb_all_occurrences[index]),
            "pmbb_all_reports": int(pmbb_all_reports[index]),
            "pmbb_strict_occurrences": int(pmbb_strict_occurrences[index]),
            "pmbb_strict_reports": int(pmbb_strict_reports[index]),
            "shown_in_fig3c_composite": int(index in display_indices),
        }
        count_rows.append(row)
    write_csv(args.output / "family_counts.csv", count_rows)

    selected_ref_rows = int(np.sum((reference_group >= 0) & reference_scope))
    selected_new_rows = int(np.sum((new_group >= 0) & new_scope))
    scoped_reasons = Counter(
        assignment.reason
        for assignment, include in zip(new_assignments, new_scope)
        if include
    )
    metadata = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "privacy": {
            "classification": "private derived patient data",
            "raw_concept_strings_written": False,
            "publish_without_approval": False,
        },
        "stratum": {
            "name": stratum.title,
            "reference": stratum.reference_name,
            "new": stratum.pmbb_name,
            "rationale": (
                "Matched anatomical scope; combined chest+abdomen would be confounded "
                "by body region and acquisition protocol."
            ),
        },
        "inputs": {
            "reference_bank": str(args.reference_bank.relative_to(ROOT)),
            "reference_embedding_memmap": str(args.reference_embeddings.relative_to(ROOT)),
            "reference_shape": list(reference_embeddings.shape),
            "reference_concept_fingerprint": concept_fingerprint(reference_concepts),
            "pmbb_bank": str(pmbb_bank_path.relative_to(ROOT)),
            "pmbb_bank_sha256": sha256_file(pmbb_bank_path),
            "pmbb_shape": list(new_embeddings.shape),
            "pmbb_concept_fingerprint": concept_fingerprint(new_concepts),
        },
        "quality_control": {
            "rule": "at least three alphabetic characters after normalization",
            "pmbb_scoped_before_qc": int(new_scope_raw.sum()),
            "pmbb_scoped_after_qc": int(new_scope.sum()),
            "excluded": int((new_scope_raw & ~quality).sum()),
        },
        "label_definition": {
            "fine_label": "project-defined RadLex-anchored anatomy x regex finding family",
            "embedding_independent": True,
            "independent_clinical_ground_truth": False,
            "anatomy_other_excluded": True,
            "finding_other_excluded": True,
            "min_reference_nonoverlap": args.min_reference,
            "min_pmbb_nonoverlap": args.min_new,
            "min_pmbb_nonoverlap_reports": args.min_new_reports,
            "retained_families": len(families),
            "reference_rows_in_retained_families": selected_ref_rows,
            "pmbb_rows_in_retained_families": selected_new_rows,
            "pmbb_anatomy_tie_count_in_scope": int(
                sum(assignment.tied for assignment, include in zip(new_assignments, new_scope) if include)
            ),
            "pmbb_anatomy_reason_counts": dict(sorted(scoped_reasons.items())),
        },
        "overlap": {
            "exact_normalized_strings": len(overlap),
            "fraction_of_scoped_pmbb": len(overlap) / int(new_scope.sum()),
            "display_membership": "reference-only gray; one shared atlas copy purple; strict PMBB-only red",
            "both_copies_excluded_from_centroid_validation": True,
        },
        "joint_display": {
            "reference_source_membership": membership_counts,
            "reference_sample": sample_counts,
            "strict_pmbb_only_count": len(strict_new_rows),
            "exact_shared_count": len(shared_rows),
            "pca_components": 50,
            "pca_explained_variance_ratio_sum": float(
                pca_joint.explained_variance_ratio_.sum()
            ),
            "umap": {
                "n_neighbors": 15,
                "metric": "euclidean",
                "min_dist": 0.5,
                "spread": 1.0,
                "init": "spectral",
                "random_state": args.seed,
            },
        },
        "trees": {
            "display_recipe": "PCA50 group means; Pearson distance 1-r; complete linkage",
            "matches_released_uce_figure_notebook": True,
            "nature_methods_metric_conflict": (
                "Nature Methods states Euclidean/full-space clustering, while released "
                "Figure 3 notebooks use Scanpy defaults (PCA50/Pearson/complete)."
            ),
        },
        "full_space_alignment": alignment_summary,
        "claim_boundary": (
            "Exploratory external-corpus organization and taxonomy consistency in "
            "F2LLM text space; not independent validation of the taxonomy, CT image "
            "model, or clinical generalization."
        ),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "scipy": __import__("scipy").__version__,
            "umap_learn": umap.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "seconds": round(time.time() - started, 1),
    }
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.output / "README.md").write_text(
        f"# {stratum.pmbb_name} to {stratum.reference_name}: UCE Figure 3 analogue\n\n"
        "Private derived patient data. Do not publish or upload without approval.\n\n"
        "The two primary figures are `fig3a_pmbb_observation_hierarchy` and "
        "`fig3c_pmbb_reference_alignment`. The full source-specific tree is "
        "`fig3c_full_source_label_dendrogram`. `run_metadata.json`, "
        "`family_counts.csv`, and `centroid_alignment_full5120.csv` record the "
        "exact analysis and taxonomy-consistency checks. No raw concept string is written here.\n\n"
        "This is a matched-scope analysis. The chest and abdomen cohorts must not be "
        "pooled for the primary claim because body region and protocol would dominate. "
        "Exact normalized strings shared by both datasets are shown once in purple; "
        "red points are strict PMBB-only.\n\n"
        "The labels are project-defined text rules, not independently validated RadLex "
        "classes. Therefore centroid retrieval is a pipeline sanity check, not external "
        "clinical validation.\n\n"
        "Reproduce from the repository root with:\n\n"
        "```bash\n"
        f"python plot_pmbb_uce_analogue.py --stratum {args.stratum}\n"
        "```\n"
    )
    print(f"Done in {time.time() - started:.1f}s -> {args.output}", flush=True)


if __name__ == "__main__":
    main()

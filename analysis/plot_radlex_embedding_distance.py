#!/usr/bin/env python3
"""Compare F2LLM concept distances with RadLex ontology-graph distances.

This is an exploratory, UCE Figure 3b-style analysis.  It deliberately uses
the full 5,120-dimensional F2LLM vectors rather than PCA or UMAP coordinates.
Concepts are grounded independently of F2LLM by conservative lexical matching
to English RadLex labels/synonyms/acronyms.

The final two-column plot balances contributions from RadLex node pairs.  The
UCE notebook's random-individual-pair sampling is retained as a tabular
sensitivity analysis without generating redundant figure variants.

Examples
--------
python plot_radlex_embedding_distance.py
python plot_radlex_embedding_distance.py --attempts 100000 --seed 42
python plot_radlex_embedding_distance.py --hash-embedding
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from lxml import etree
from scipy.stats import gaussian_kde, spearmanr, ttest_ind

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_BANK = ROOT / "concept_bank.f2llm_emb.npz"
DEFAULT_EMBEDDINGS = ROOT / "_emb_f2llm_tmp.npy"
DEFAULT_RADLEX = ROOT / "data" / "radlex" / "4.3" / "RadLex.owl"
DEFAULT_OUT_DIR = ROOT / "outputs" / "concept_ontology" / "f2llm"

BRANCHES = {
    "RID5": "Imaging observation",
    "RID34785": "Clinical finding",
}

UCE_BRANCH_TITLES = {
    "RID5": "Imaging observations",
    "RID34785": "Clinical finding",
}

# Keep each branch within one hue while making ontology hops visually distinct.
# Hop 1 is the lightest shade and the maximum plotted hop is the darkest.
BRANCH_BASE_COLORS = {
    "RID34785": "#2F67B1",  # Clinical finding: blue.
    "RID5": "#7651A5",  # Imaging observations: violet.
}
DISTRIBUTION_LINE_COLOR = "#000000"
FINAL_FIGURE_STEM = "radlex_f2llm_distance_ridges_balanced"
PLOT_BRANCH_ORDER = ("RID34785", "RID5")
LEGACY_FIGURE_STEMS = (
    "radlex_f2llm_distance_ridges_uce_style",
    "radlex_f2llm_distance_rid34785_balanced_uce_layout",
    "radlex_f2llm_distance_rid34785_uce_style_layout",
    "radlex_f2llm_distance_rid5_balanced_uce_layout",
    "radlex_f2llm_distance_rid5_uce_style_layout",
)

KDE_BW_ADJUST = 0.5

# Figure typography, sized for readability after the two-column panel is placed
# into a manuscript page. These cover the elements highlighted in the figure
# review: axes, hop/sample labels, and inter-row significance annotations.
AXIS_LABEL_FONTSIZE = 22
AXIS_TICK_FONTSIZE = 15
HOP_LABEL_FONTSIZE = 20
SAMPLE_SIZE_FONTSIZE = 18
PVALUE_FONTSIZE = 12
PANEL_TITLE_FONTSIZE = 21

OWL_NS = "http://www.w3.org/2002/07/owl#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
RADLEX_NS = "http://www.radlex.org/RID/"
XML_NS = "http://www.w3.org/XML/1998/namespace"
RDF_ABOUT = f"{{{RDF_NS}}}about"
RDF_RESOURCE = f"{{{RDF_NS}}}resource"
RDF_ROOT = f"{{{RDF_NS}}}RDF"
XML_LANG = f"{{{XML_NS}}}lang"
RDFS_LABEL = f"{{{RDFS_NS}}}label"
RDFS_SUBCLASS_OF = f"{{{RDFS_NS}}}subClassOf"
RADLEX_SYNONYM = f"{{{RADLEX_NS}}}Synonym"
RADLEX_ACRONYM = f"{{{RADLEX_NS}}}Acronym"
RADLEX_OBSOLETE_NAME = f"{{{RADLEX_NS}}}Preferred_Name_for_Obsolete"


@dataclass
class RadLexData:
    hierarchy: nx.DiGraph
    preferred_labels: dict[str, str]
    terms: dict[str, set[str]]
    obsolete: set[str]


def _rid_from_uri(uri: str | None) -> str | None:
    if not uri or "/RID/" not in uri:
        return None
    rid = uri.rsplit("/", 1)[-1]
    return rid if rid.startswith("RID") else None


def _english(element: etree._Element) -> bool:
    return element.get(XML_LANG) in (None, "en")


def _collect_radlex_annotations(
    element: etree._Element,
    rid: str,
    preferred: dict[str, str],
    terms: dict[str, set[str]],
    obsolete: set[str],
) -> None:
    """Merge English lexical annotations from one RDF/XML fragment."""
    english_labels: list[str] = []
    for child in element:
        if child.tag == RDFS_LABEL and child.text and _english(child):
            label = child.text.strip()
            if label:
                english_labels.append(label)
                terms[rid].add(label)
        elif (
            child.tag in {RADLEX_SYNONYM, RADLEX_ACRONYM}
            and child.text
            and _english(child)
        ):
            term = child.text.strip()
            if term:
                terms[rid].add(term)
        elif child.tag == RADLEX_OBSOLETE_NAME:
            obsolete.add(rid)
    if english_labels:
        preferred.setdefault(rid, english_labels[0])


def _clear_parsed_element(element: etree._Element) -> None:
    """Release an iterparse element and its processed siblings."""
    element.clear()
    parent_element = element.getparent()
    if parent_element is not None:
        while element.getprevious() is not None:
            del parent_element[0]


def parse_radlex(path: Path) -> RadLexData:
    """Parse direct subclass edges and merge split RadLex annotations by RID.

    RadLex 4.3 serializes many class annotations in separate ``rdf:Description``
    elements rather than inside the corresponding ``owl:Class`` element.  The
    hierarchy is therefore read first from classes, followed by a second pass
    that merges lexical annotations from description fragments for known class
    RIDs.
    """
    hierarchy = nx.DiGraph()
    preferred: dict[str, str] = {}
    terms: dict[str, set[str]] = defaultdict(set)
    obsolete: set[str] = set()

    for _, element in etree.iterparse(
        str(path), events=("end",), tag=f"{{{OWL_NS}}}Class"
    ):
        rid = _rid_from_uri(element.get(RDF_ABOUT))
        if rid:
            hierarchy.add_node(rid)
            for child in element:
                if child.tag == RDFS_SUBCLASS_OF:
                    parent = _rid_from_uri(child.get(RDF_RESOURCE))
                    if parent:
                        hierarchy.add_edge(parent, rid)
            _collect_radlex_annotations(element, rid, preferred, terms, obsolete)

        _clear_parsed_element(element)

    class_rids = set(hierarchy.nodes)
    for _, element in etree.iterparse(
        str(path), events=("end",), tag=f"{{{RDF_NS}}}Description"
    ):
        rid = _rid_from_uri(element.get(RDF_ABOUT))
        parent_element = element.getparent()
        is_top_level = parent_element is not None and parent_element.tag == RDF_ROOT
        if is_top_level and rid in class_rids:
            _collect_radlex_annotations(element, rid, preferred, terms, obsolete)
        _clear_parsed_element(element)

    return RadLexData(
        hierarchy=hierarchy,
        preferred_labels=preferred,
        terms=dict(terms),
        obsolete=obsolete,
    )


def tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.lower()))


def normalized(text: str) -> str:
    return " ".join(tokens(text))


def branch_term_index(
    radlex: RadLexData,
    root_rid: str,
) -> tuple[nx.Graph, set[str], dict[tuple[str, ...], str], dict[str, int]]:
    """Return branch graph and unambiguous normalized term-to-RID index."""
    branch_nodes = nx.descendants(radlex.hierarchy, root_rid) | {root_rid}
    graph = radlex.hierarchy.subgraph(branch_nodes).to_undirected().copy()

    active_nodes = branch_nodes - radlex.obsolete - {root_rid}
    term_nodes: set[str] = set()
    term_to_rids: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for rid in active_nodes:
        for term in radlex.terms.get(rid, set()):
            key = tokens(term)
            if key:
                term_nodes.add(rid)
                term_to_rids[key].add(rid)

    unique = {
        term: next(iter(rids))
        for term, rids in term_to_rids.items()
        if len(rids) == 1
    }
    audit = {
        "branch_nodes": len(branch_nodes),
        "active_nonroot_nodes": len(active_nodes),
        "active_term_nodes": len(term_nodes),
        "active_nodes_without_terms": len(active_nodes - term_nodes),
        "normalized_terms": len(term_to_rids),
        "unambiguous_terms": len(unique),
        "ambiguous_terms_excluded": sum(
            len(rids) > 1 for rids in term_to_rids.values()
        ),
        "obsolete_nodes_excluded_from_mapping": len(branch_nodes & radlex.obsolete),
    }
    return graph, branch_nodes, unique, audit


def map_concepts(
    concepts: np.ndarray,
    radlex: RadLexData,
    root_rid: str,
) -> tuple[pd.DataFrame, dict[str, int], nx.Graph, dict[str, int]]:
    """Conservatively ground concepts to one RadLex RID using exact token spans."""
    graph, _, term_index, ontology_audit = branch_term_index(radlex, root_rid)
    terms_by_first: dict[str, list[tuple[tuple[str, ...], str]]] = defaultdict(list)
    for term, rid in term_index.items():
        terms_by_first[term[0]].append((term, rid))
    for candidates in terms_by_first.values():
        candidates.sort(key=lambda item: len(item[0]), reverse=True)

    counts: Counter[str] = Counter()
    records: list[dict[str, object]] = []

    for concept_index, concept_value in enumerate(concepts):
        concept = str(concept_value)
        concept_tokens = tokens(concept)
        if not concept_tokens:
            counts["no_ontology_term"] += 1
            continue

        matches: set[tuple[int, int, str, tuple[str, ...]]] = set()
        for start, first_token in enumerate(concept_tokens):
            for term, rid in terms_by_first.get(first_token, []):
                end = start + len(term)
                if end <= len(concept_tokens) and concept_tokens[start:end] == term:
                    matches.add((start, end, rid, term))

        if not matches:
            counts["no_ontology_term"] += 1
            continue

        # Keep maximal token spans.  Shorter contained terms such as "opacity"
        # inside "ground glass opacity" must not create artificial ambiguity.
        maximal = []
        for match in matches:
            start, end, _, _ = match
            contained = any(
                other[0] <= start
                and other[1] >= end
                and (other[0] < start or other[1] > end)
                for other in matches
            )
            if not contained:
                maximal.append(match)

        matched_rids = {match[2] for match in maximal}
        if len(matched_rids) != 1:
            counts["multiple_rid_ambiguity"] += 1
            continue

        safe_multiword = any((end - start) >= 2 for start, end, _, _ in maximal)
        safe_single_exact = (
            len(concept_tokens) == 1
            and any(start == 0 and end == 1 for start, end, _, _ in maximal)
        )
        if not (safe_multiword or safe_single_exact):
            counts["unsafe_single_token_in_clause"] += 1
            continue

        rid = next(iter(matched_rids))
        maximal.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[3]))
        matched_terms = [" ".join(match[3]) for match in maximal]
        whole_phrase = (
            len(maximal) == 1
            and maximal[0][0] == 0
            and maximal[0][1] == len(concept_tokens)
        )
        records.append(
            {
                "concept_index": concept_index,
                "concept": concept,
                "rid": rid,
                "rid_label": radlex.preferred_labels.get(rid, rid),
                "matched_terms": " | ".join(matched_terms),
                "match_mode": "whole_phrase" if whole_phrase else "multiword_span",
                "max_matched_tokens": max(end - start for start, end, _, _ in maximal),
            }
        )
        counts["accepted"] += 1

    for key in (
        "accepted",
        "no_ontology_term",
        "multiple_rid_ambiguity",
        "unsafe_single_token_in_clause",
    ):
        counts.setdefault(key, 0)
    if sum(counts.values()) != len(concepts):
        raise AssertionError("mapping outcome counts do not cover every concept")

    mapping = pd.DataFrame.from_records(records)
    if not mapping.empty:
        mapping = mapping.sort_values("concept_index").reset_index(drop=True)
    return mapping, dict(counts), graph, ontology_audit


def ontology_distances(
    graph: nx.Graph,
    rids: Iterable[str],
) -> tuple[dict[tuple[str, str], int], dict[int, list[tuple[str, str]]]]:
    """Precompute distances between mapped nodes and index unordered pairs by hop."""
    unique_rids = sorted(set(rids))
    rid_set = set(unique_rids)
    distances: dict[tuple[str, str], int] = {}
    pairs_by_hop: dict[int, list[tuple[str, str]]] = defaultdict(list)

    for left_index, left in enumerate(unique_rids):
        lengths = nx.single_source_shortest_path_length(graph, left)
        for right in unique_rids[left_index + 1 :]:
            if right not in rid_set or right not in lengths:
                continue
            hop = int(lengths[right])
            pair = (left, right)
            distances[pair] = hop
            pairs_by_hop[hop].append(pair)

    return distances, dict(pairs_by_hop)


def _ordered_rid_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def sample_uce_style(
    mapping: pd.DataFrame,
    distance_lookup: dict[tuple[str, str], int],
    attempts: int,
    seed: int,
) -> pd.DataFrame:
    """Mirror UCE: fixed attempts, distinct rows, skip equal ontology labels."""
    if len(mapping) < 2:
        raise ValueError("at least two mapped concepts are required")
    rng = random.Random(seed)
    concept_indices = mapping["concept_index"].to_numpy(np.int64)
    rids = mapping["rid"].astype(str).to_numpy()
    records: list[tuple[int, int, int, str, str, int]] = []

    for attempt_index in range(attempts):
        left_local, right_local = rng.sample(range(len(mapping)), 2)
        left_rid = rids[left_local]
        right_rid = rids[right_local]
        if left_rid == right_rid:
            continue
        ordered = _ordered_rid_pair(left_rid, right_rid)
        hop = distance_lookup.get(ordered)
        if hop is None:
            continue
        records.append(
            (
                attempt_index,
                int(concept_indices[left_local]),
                int(concept_indices[right_local]),
                left_rid,
                right_rid,
                int(hop),
            )
        )

    return pd.DataFrame.from_records(
        records,
        columns=(
            "attempt_index",
            "left_concept_index",
            "right_concept_index",
            "left_rid",
            "right_rid",
            "ontology_distance",
        ),
    )


def _sample_unique_flat_indices(
    total: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a small number of unique integers without allocating total entries."""
    count = min(count, total)
    if count == total:
        return np.arange(total, dtype=np.int64)
    selected: set[int] = set()
    while len(selected) < count:
        selected.add(int(rng.integers(0, total)))
    return np.fromiter(sorted(selected), dtype=np.int64)


def sample_rid_pair_balanced(
    mapping: pd.DataFrame,
    pairs_by_hop: dict[int, list[tuple[str, str]]],
    max_hop: int,
    max_rid_pairs_per_hop: int,
    max_phrase_pairs_per_rid_pair: int,
    seed: int,
) -> pd.DataFrame:
    """Cap every RID-pair contribution and sample RID pairs uniformly per hop."""
    rng = np.random.default_rng(seed)
    rows_by_rid = {
        rid: group["concept_index"].to_numpy(np.int64)
        for rid, group in mapping.groupby("rid", sort=True)
    }
    records: list[tuple[int, int, str, str, int]] = []

    for hop in range(1, max_hop + 1):
        rid_pairs = sorted(pairs_by_hop.get(hop, []))
        if len(rid_pairs) > max_rid_pairs_per_hop:
            chosen = rng.choice(
                len(rid_pairs), size=max_rid_pairs_per_hop, replace=False
            )
            rid_pairs = [rid_pairs[index] for index in sorted(chosen)]

        for left_rid, right_rid in rid_pairs:
            left_rows = rows_by_rid[left_rid]
            right_rows = rows_by_rid[right_rid]
            total = len(left_rows) * len(right_rows)
            flat = _sample_unique_flat_indices(
                total,
                max_phrase_pairs_per_rid_pair,
                rng,
            )
            right_count = len(right_rows)
            for value in flat:
                left_row = int(left_rows[int(value // right_count)])
                right_row = int(right_rows[int(value % right_count)])
                records.append((left_row, right_row, left_rid, right_rid, hop))

    return pd.DataFrame.from_records(
        records,
        columns=(
            "left_concept_index",
            "right_concept_index",
            "left_rid",
            "right_rid",
            "ontology_distance",
        ),
    )


def add_embedding_distances(
    pairs: pd.DataFrame,
    embeddings: np.memmap,
    block_size: int,
) -> pd.DataFrame:
    """Add normalized Euclidean distances without materializing the full bank."""
    left = pairs["left_concept_index"].to_numpy(np.int64)
    right = pairs["right_concept_index"].to_numpy(np.int64)
    distances = np.empty(len(pairs), dtype=np.float32)

    for start in range(0, len(pairs), block_size):
        stop = min(start + block_size, len(pairs))
        a = np.asarray(embeddings[left[start:stop]], dtype=np.float32)
        b = np.asarray(embeddings[right[start:stop]], dtype=np.float32)
        aa = np.einsum("ij,ij->i", a, a)
        bb = np.einsum("ij,ij->i", b, b)
        ab = np.einsum("ij,ij->i", a, b)
        denominator = np.sqrt(np.maximum(aa * bb, 1e-24))
        cosine = np.clip(ab / denominator, -1.0, 1.0)
        distances[start:stop] = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine))

    if not np.isfinite(distances).all():
        raise ValueError("non-finite embedding distances encountered")
    result = pairs.copy()
    result["embedding_distance"] = distances
    return result


def _holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(pvalues[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    means = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = rng.choice(values, size=len(values), replace=True)
        means[index] = float(np.mean(sample))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_pairs(
    pairs: pd.DataFrame,
    max_hop: int,
    seed: int,
    bootstrap_replicates: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    subset = pairs[pairs["ontology_distance"].between(1, max_hop)].copy()
    if subset.empty:
        raise ValueError("no pairs at requested ontology distances")
    subset["rid_pair"] = [
        "||".join(_ordered_rid_pair(left, right))
        for left, right in zip(subset["left_rid"], subset["right_rid"])
    ]

    rng = np.random.default_rng(seed)
    summaries: list[dict[str, object]] = []
    node_pair_means_by_hop: dict[int, np.ndarray] = {}
    for hop in range(1, max_hop + 1):
        hop_data = subset[subset["ontology_distance"] == hop]
        values = hop_data["embedding_distance"].to_numpy(float)
        node_means = (
            hop_data.groupby("rid_pair", sort=True)["embedding_distance"]
            .mean()
            .to_numpy(float)
        )
        node_pair_means_by_hop[hop] = node_means
        ci_low, ci_high = _bootstrap_mean_ci(
            node_means, rng, bootstrap_replicates
        )
        summaries.append(
            {
                "ontology_distance": hop,
                "phrase_pairs": len(values),
                "rid_pairs": len(node_means),
                "mean": float(np.mean(values)) if len(values) else float("nan"),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                "median": float(np.median(values)) if len(values) else float("nan"),
                "q25": float(np.quantile(values, 0.25)) if len(values) else float("nan"),
                "q75": float(np.quantile(values, 0.75)) if len(values) else float("nan"),
                "rid_pair_mean": float(np.mean(node_means)) if len(node_means) else float("nan"),
                "rid_pair_bootstrap_ci_low": ci_low,
                "rid_pair_bootstrap_ci_high": ci_high,
            }
        )

    adjacent: list[dict[str, object]] = []
    phrase_pvalues: list[float] = []
    node_pvalues: list[float] = []
    for hop in range(1, max_hop):
        current = subset.loc[
            subset["ontology_distance"] == hop, "embedding_distance"
        ].to_numpy(float)
        following = subset.loc[
            subset["ontology_distance"] == hop + 1, "embedding_distance"
        ].to_numpy(float)
        current_nodes = node_pair_means_by_hop[hop]
        following_nodes = node_pair_means_by_hop[hop + 1]

        phrase_test = ttest_ind(
            current,
            following,
            equal_var=True,
            alternative="less",
        )
        node_test = ttest_ind(
            current_nodes,
            following_nodes,
            equal_var=False,
            alternative="less",
        )
        phrase_pvalues.append(float(phrase_test.pvalue))
        node_pvalues.append(float(node_test.pvalue))
        adjacent.append(
            {
                "lower_hop": hop,
                "upper_hop": hop + 1,
                "mean_difference_upper_minus_lower": float(
                    np.mean(following) - np.mean(current)
                ),
                "uce_style_phrase_t": float(phrase_test.statistic),
                "uce_style_phrase_p_one_sided": float(phrase_test.pvalue),
                "rid_pair_welch_t": float(node_test.statistic),
                "rid_pair_welch_p_one_sided": float(node_test.pvalue),
            }
        )

    phrase_adjusted = _holm_adjust(phrase_pvalues)
    node_adjusted = _holm_adjust(node_pvalues)
    for result, phrase_p, node_p in zip(adjacent, phrase_adjusted, node_adjusted):
        result["uce_style_phrase_p_holm"] = phrase_p
        result["rid_pair_welch_p_holm"] = node_p

    rho = spearmanr(
        subset["ontology_distance"].to_numpy(float),
        subset["embedding_distance"].to_numpy(float),
    )
    overall = {
        "included_phrase_pairs": len(subset),
        "included_rid_pairs": int(subset["rid_pair"].nunique()),
        "spearman_rho": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        "adjacent_tests": adjacent,
    }
    return pd.DataFrame.from_records(summaries), overall


def _branch_hop_colors(root_rid: str, max_hop: int) -> dict[int, tuple[float, ...]]:
    """Return light-to-dark shades of the branch hue for hops 1 to max_hop."""
    if max_hop < 1:
        raise ValueError("max_hop must be at least 1")
    base_color = BRANCH_BASE_COLORS[root_rid]
    shades = sns.light_palette(base_color, n_colors=max_hop + 2)[2:]
    return {hop: tuple(shades[hop - 1]) for hop in range(1, max_hop + 1)}


def _draw_embedding_distance_distribution(
    ax: plt.Axes,
    values: np.ndarray,
    fill_color: tuple[float, ...],
    line_color: str,
    alpha: float,
    clip: tuple[float, float],
) -> None:
    """Draw one ridge and a dashed vertical segment at its sample mean."""
    if len(values) >= 3 and np.ptp(values) > 0:
        sns.kdeplot(
            x=values,
            ax=ax,
            bw_adjust=KDE_BW_ADJUST,
            fill=True,
            color=fill_color,
            alpha=alpha,
            linewidth=1.35,
            edgecolor=line_color,
            clip=clip,
            cut=0,
        )
        mean_value = float(np.mean(values))
        density = gaussian_kde(values)
        density.set_bandwidth(density.factor * KDE_BW_ADJUST)
        mean_height = float(density(mean_value)[0])
        ax.vlines(
            mean_value,
            0.0,
            mean_height,
            color=line_color,
            linestyle=(0, (5, 3)),
            linewidth=1.6,
            zorder=4,
        )
    elif len(values):
        ax.hist(
            values,
            bins=15,
            density=True,
            color=fill_color,
            edgecolor=line_color,
            linewidth=1.35,
            alpha=alpha,
        )
        ax.axvline(
            float(np.mean(values)),
            color=line_color,
            linestyle=(0, (5, 3)),
            linewidth=1.6,
            zorder=4,
        )


def _format_rotated_pvalue(pvalue: float) -> str:
    """Format an adjusted p-value compactly for the vertical bracket label."""
    if not math.isfinite(pvalue):
        return "p = NA"
    return f"p = {pvalue:.4g}"


def plot_uce_two_branch(
    pair_tables: dict[str, pd.DataFrame],
    statistics: dict[str, dict[str, object]],
    output_stem: Path,
    max_hop: int,
    dpi: int,
) -> None:
    """Render the final two-branch figure in the visual grammar of UCE Figure 3b."""
    row_order = list(range(max_hop, 0, -1))
    sns.set_theme(style="white")
    fig = plt.figure(figsize=(14.0, 8.4))
    grid = fig.add_gridspec(
        max_hop,
        2 * len(PLOT_BRANCH_ORDER),
        height_ratios=([1.0] * max_hop),
        width_ratios=(5.4, 0.70, 5.4, 0.70),
        left=0.07,
        right=0.985,
        bottom=0.09,
        top=0.94,
        hspace=0.08,
        wspace=0.06,
    )

    for column, root_rid in enumerate(PLOT_BRANCH_ORDER):
        pairs = pair_tables[root_rid]
        adjacent = {
            int(item["lower_hop"]): float(item["rid_pair_welch_p_holm"])
            for item in statistics[root_rid]["overall"]["adjacent_tests"]
        }
        hop_colors = _branch_hop_colors(root_rid, max_hop)
        ridge_column = 2 * column
        significance_column = ridge_column + 1
        ridge_axes: dict[int, plt.Axes] = {}

        for row, hop in enumerate(row_order):
            share_axis = ridge_axes.get(max_hop)
            ax = fig.add_subplot(grid[row, ridge_column], sharex=share_axis)
            ridge_axes[hop] = ax
            hop_data = pairs.loc[
                pairs["ontology_distance"] == hop, "embedding_distance"
            ].to_numpy(float)
            _draw_embedding_distance_distribution(
                ax,
                hop_data,
                fill_color=hop_colors[hop],
                line_color=DISTRIBUTION_LINE_COLOR,
                alpha=0.86,
                clip=(0.0, 1.5),
            )
            ax.axhline(0, color="#303030", linewidth=1.0)
            ax.set_xlim(0.0, 1.5)
            ax.set_yticks([])
            ax.set_ylabel("")
            for spine in ("left", "right", "top"):
                ax.spines[spine].set_visible(False)
            if hop != 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            else:
                ax.set_xlabel(
                    "Embedding distance in F2LLM",
                    fontsize=AXIS_LABEL_FONTSIZE,
                    labelpad=10,
                )
                ax.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE)
            ax.text(
                0.008,
                0.03,
                str(hop),
                transform=ax.transAxes,
                fontsize=HOP_LABEL_FONTSIZE,
                ha="left",
                va="bottom",
            )
            ax.text(
                0.03,
                0.82,
                f"($n$ = {len(hop_data):,})",
                transform=ax.transAxes,
                fontsize=SAMPLE_SIZE_FONTSIZE,
                ha="left",
                va="top",
            )

            significance_ax = fig.add_subplot(grid[row, significance_column])
            significance_ax.axis("off")
            if hop < max_hop:
                pvalue = adjacent.get(hop, float("nan"))
                significance_ax.plot(
                    [0.05, 0.55, 0.55, 0.05],
                    [0.12, 0.12, 1.08, 1.08],
                    transform=significance_ax.transAxes,
                    color="#444444",
                    linewidth=1.0,
                    clip_on=False,
                )
                significance_ax.text(
                    0.72,
                    0.60,
                    _format_rotated_pvalue(pvalue),
                    transform=significance_ax.transAxes,
                    rotation=90,
                    fontsize=PVALUE_FONTSIZE,
                    ha="center",
                    va="center",
                    clip_on=False,
                )

        ridge_axes[max_hop].set_title(
            UCE_BRANCH_TITLES[root_rid],
            fontsize=PANEL_TITLE_FONTSIZE,
            fontweight="bold",
            pad=15,
        )

    fig.text(
        0.026,
        0.56,
        "Shortest-path distance in RadLex",
        rotation=90,
        va="center",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)


def remove_legacy_figure_files(out_dir: Path) -> list[Path]:
    """Remove superseded plot variants while retaining all analysis artifacts."""
    removed: list[Path] = []
    for stem in LEGACY_FIGURE_STEMS:
        for suffix in (".png", ".pdf"):
            path = out_dir / f"{stem}{suffix}"
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def concept_fingerprint(concepts: np.ndarray) -> str:
    digest = hashlib.sha256()
    for concept in concepts:
        digest.update(str(concept).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--embedding-npy", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--radlex-owl", type=Path, default=DEFAULT_RADLEX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--attempts", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hop", type=int, default=5)
    parser.add_argument("--max-rid-pairs-per-hop", type=int, default=200)
    parser.add_argument("--max-phrase-pairs-per-rid-pair", type=int, default=100)
    parser.add_argument("--distance-block-size", type=int, default=2_048)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--hash-embedding",
        action="store_true",
        help="stream the 7.7-GB embedding file to record its SHA256",
    )
    args = parser.parse_args()

    for name in (
        "attempts",
        "max_hop",
        "max_rid_pairs_per_hop",
        "max_phrase_pairs_per_rid_pair",
        "distance_block_size",
        "bootstrap_replicates",
        "dpi",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.bank, allow_pickle=True) as bank:
        concepts = bank["concepts"]
    embeddings = np.load(args.embedding_npy, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(concepts):
        raise ValueError(
            f"row mismatch: concepts={len(concepts):,}, embeddings={embeddings.shape}"
        )
    if embeddings.dtype != np.float32:
        raise ValueError(f"expected float32 embeddings, found {embeddings.dtype}")

    print(f"concepts: {len(concepts):,}")
    print(f"embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}, mmap=True")
    print(f"parsing {args.radlex_owl}")
    radlex = parse_radlex(args.radlex_owl)

    balanced_tables: dict[str, pd.DataFrame] = {}
    mapping_audits: dict[str, object] = {}
    all_statistics: dict[str, object] = {"balanced": {}, "uce_style": {}}
    hop_summaries: list[pd.DataFrame] = []

    for branch_index, (root_rid, branch_label) in enumerate(BRANCHES.items()):
        print(f"\n[{root_rid}] mapping {branch_label.lower()} concepts")
        mapping, outcome_counts, graph, ontology_audit = map_concepts(
            concepts,
            radlex,
            root_rid,
        )
        if mapping.empty:
            raise ValueError(f"no concepts mapped under {root_rid}")
        mapping.to_csv(
            args.out_dir / f"mapping_{root_rid.lower()}.csv.gz",
            index=False,
            compression="gzip",
        )
        print(
            f"accepted {len(mapping):,} concepts across {mapping['rid'].nunique():,} RIDs; "
            + ", ".join(f"{key}={value:,}" for key, value in outcome_counts.items())
        )

        distance_lookup, pairs_by_hop = ontology_distances(graph, mapping["rid"])
        node_pair_counts = {
            str(hop): len(pairs_by_hop.get(hop, []))
            for hop in range(1, args.max_hop + 1)
        }
        mapping_audits[root_rid] = {
            "branch_label": branch_label,
            "outcomes": outcome_counts,
            "accepted_unique_rids": int(mapping["rid"].nunique()),
            "ontology_index": ontology_audit,
            "available_rid_pairs_by_hop": node_pair_counts,
        }

        branch_seed = args.seed + branch_index * 10_000
        print(f"[{root_rid}] sampling RID-pair-balanced pairs")
        balanced = sample_rid_pair_balanced(
            mapping,
            pairs_by_hop,
            max_hop=args.max_hop,
            max_rid_pairs_per_hop=args.max_rid_pairs_per_hop,
            max_phrase_pairs_per_rid_pair=args.max_phrase_pairs_per_rid_pair,
            seed=branch_seed,
        )
        balanced = add_embedding_distances(
            balanced,
            embeddings,
            block_size=args.distance_block_size,
        )
        balanced["branch_root"] = root_rid
        balanced["sampling"] = "rid_pair_balanced"
        balanced_tables[root_rid] = balanced
        balanced.to_csv(
            args.out_dir / f"pairs_balanced_{root_rid.lower()}.csv.gz",
            index=False,
            compression="gzip",
        )

        print(f"[{root_rid}] sampling {args.attempts:,} UCE-style attempts")
        uce_style = sample_uce_style(
            mapping,
            distance_lookup,
            attempts=args.attempts,
            seed=branch_seed,
        )
        uce_style = add_embedding_distances(
            uce_style,
            embeddings,
            block_size=args.distance_block_size,
        )
        uce_style["branch_root"] = root_rid
        uce_style["sampling"] = "uce_style_random_phrase_pairs"
        uce_style.to_csv(
            args.out_dir / f"pairs_uce_style_{root_rid.lower()}.csv.gz",
            index=False,
            compression="gzip",
        )

        for mode, table in (("balanced", balanced), ("uce_style", uce_style)):
            summary, overall = summarize_pairs(
                table,
                max_hop=args.max_hop,
                seed=branch_seed + (0 if mode == "balanced" else 1),
                bootstrap_replicates=args.bootstrap_replicates,
            )
            summary.insert(0, "branch_root", root_rid)
            summary.insert(1, "branch_label", branch_label)
            summary.insert(2, "sampling", mode)
            hop_summaries.append(summary)
            all_statistics[mode][root_rid] = {
                "overall": overall,
                "by_hop": summary.to_dict(orient="records"),
            }
            print(
                f"[{root_rid}] {mode}: pairs at hops 1-{args.max_hop}="
                f"{overall['included_phrase_pairs']:,}, rho={overall['spearman_rho']:.4f}"
            )

    plot_uce_two_branch(
        balanced_tables,
        all_statistics["balanced"],
        output_stem=args.out_dir / FINAL_FIGURE_STEM,
        max_hop=args.max_hop,
        dpi=args.dpi,
    )
    removed_figures = remove_legacy_figure_files(args.out_dir)
    if removed_figures:
        print(f"Removed {len(removed_figures)} superseded figure files")

    pd.concat(hop_summaries, ignore_index=True).to_csv(
        args.out_dir / "distance_summary_by_hop.csv",
        index=False,
    )
    with (args.out_dir / "statistics.json").open("w") as handle:
        json.dump(_json_safe(all_statistics), handle, indent=2)
        handle.write("\n")
    with (args.out_dir / "mapping_audit.json").open("w") as handle:
        json.dump(_json_safe(mapping_audits), handle, indent=2)
        handle.write("\n")

    metadata = {
        "analysis": "UCE Figure 3b-style RadLex graph-distance analysis",
        "claim_boundary": (
            "Exploratory lexical grounding of report-derived phrases; not a manually "
            "validated concept-to-RadLex annotation and not an image-model analysis."
        ),
        "parameters": vars(args),
        "inputs": {
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "concept_bank": str(args.bank.resolve()),
            "concept_bank_bytes": args.bank.stat().st_size,
            "concept_count": len(concepts),
            "concept_order_sha256": concept_fingerprint(concepts),
            "embedding_npy": str(args.embedding_npy.resolve()),
            "embedding_bytes": args.embedding_npy.stat().st_size,
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
            "embedding_sha256": (
                sha256_file(args.embedding_npy) if args.hash_embedding else None
            ),
            "radlex_owl": str(args.radlex_owl.resolve()),
            "radlex_owl_bytes": args.radlex_owl.stat().st_size,
            "radlex_owl_sha256": sha256_file(args.radlex_owl),
        },
        "radlex_parsing": {
            "hierarchy": "direct named rdfs:subClassOf edges from owl:Class",
            "annotations": (
                "English or language-unspecified rdfs:label, Synonym, and Acronym "
                "annotations merged by RID from owl:Class and top-level rdf:Description"
            ),
            "obsolete_marker": "Preferred_Name_for_Obsolete from both serializations",
            "branches": BRANCHES,
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "seaborn": sns.__version__,
            "matplotlib": matplotlib.__version__,
            "networkx": nx.__version__,
            "lxml": etree.LXML_VERSION,
        },
    }
    with (args.out_dir / "analysis_metadata.json").open("w") as handle:
        json.dump(_json_safe(metadata), handle, indent=2, default=str)
        handle.write("\n")

    print("\nWrote:")
    for path in sorted(args.out_dir.iterdir()):
        try:
            display_path = path.relative_to(ROOT)
        except ValueError:
            display_path = path
        print(f"  {display_path}")


if __name__ == "__main__":
    main()

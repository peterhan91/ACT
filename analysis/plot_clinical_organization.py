#!/usr/bin/env python3
"""Plot clinical organization and a same-coordinate source-dataset control.

The main atlas colors the cached F2LLM UMAP by the project-defined,
RadLex-anchored anatomical families.  The companion source panel uses the
*same rows and same UMAP coordinates* but colors a balanced random sample by
the JSONL in which each unique observation phrase occurred.

Each row is one exact-deduplicated, lower-cased observation phrase.  It is not
a scan, patient, report, or frequency-weighted observation.  Phrases that
occur verbatim in both corpora are reported but omitted from the two-source
balanced sample so they are not assigned arbitrarily to either cohort.

Usage:
    python plot_clinical_organization.py
    python plot_clinical_organization.py --sample-per-source 40000 --seed 42
    python plot_clinical_organization.py --out /tmp/clinical_organization.pdf
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from radlex_anatomy_categories import (  # noqa: E402
    ANATOMY_COLORS,
    GROUPS as ANATOMY_GROUPS,
    OTHER,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BANK = ROOT / "concept_bank.f2llm_emb.npz"
DEFAULT_UMAP_DIR = ROOT / "outputs" / "concept_umap" / "f2llm"
DEFAULT_OUT = DEFAULT_UMAP_DIR / "clinical_organization_source_inset.png"
DEFAULT_SOURCE_ONLY_OUT = DEFAULT_UMAP_DIR / "concept_source_dataset_all.png"
DEFAULT_CTRATE = ROOT / "ctrate_concepts.jsonl"
DEFAULT_MERLIN = ROOT / "merlin_concepts.jsonl"

CTR_ONLY = np.uint8(0)
MERLIN_ONLY = np.uint8(1)
SHARED = np.uint8(2)

# Okabe-Ito blue and vermillion: colorblind-safe and balanced in visual weight.
SOURCE_COLORS = {
    CTR_ONLY: "#0072B2",
    MERLIN_ONLY: "#D55E00",
    SHARED: "#7A5195",
}


def _observations(path: Path) -> set[str]:
    """Reconstruct the exact phrase normalization used by get_embed_ct.py."""
    phrases: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            raw = record.get("model_output", "")
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                try:
                    payload = ast.literal_eval(raw)
                except Exception:
                    # The production bank also drops unparseable model output.
                    continue
            observations = payload.get("observations") or []
            if not isinstance(observations, list):
                raise ValueError(
                    f"{path}:{line_number}: observations is not a list"
                )
            for observation in observations:
                if isinstance(observation, str):
                    phrase = observation.lower().strip()
                    if phrase:
                        phrases.add(phrase)
    return phrases


def source_membership(
    concepts: np.ndarray,
    ctrate_jsonl: Path,
    merlin_jsonl: Path,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return row-aligned CT-only/Merlin-only/shared source membership."""
    ctrate = _observations(ctrate_jsonl)
    merlin = _observations(merlin_jsonl)
    union = ctrate | merlin
    bank = set(map(str, concepts))
    if bank != union:
        raise ValueError(
            "concept bank does not match the JSONL union: "
            f"missing={len(union - bank):,}, extra={len(bank - union):,}"
        )

    membership = np.empty(len(concepts), dtype=np.uint8)
    for i, concept in enumerate(concepts):
        in_ctrate = concept in ctrate
        in_merlin = concept in merlin
        if in_ctrate and in_merlin:
            membership[i] = SHARED
        elif in_ctrate:
            membership[i] = CTR_ONLY
        elif in_merlin:
            membership[i] = MERLIN_ONLY
        else:  # protected by the exact-union check above
            raise AssertionError(f"unmapped concept row {i}: {concept!r}")

    counts = {
        "CT-RATE only": int((membership == CTR_ONLY).sum()),
        "Merlin only": int((membership == MERLIN_ONLY).sum()),
        "Shared exact phrase": int((membership == SHARED).sum()),
    }
    return membership, counts


def balanced_source_sample(
    membership: np.ndarray,
    sample_per_source: int,
    seed: int,
) -> np.ndarray:
    """Sample source-exclusive rows equally, then randomize their draw order."""
    rng = np.random.default_rng(seed)
    sampled = []
    for source in (CTR_ONLY, MERLIN_ONLY):
        candidates = np.flatnonzero(membership == source)
        n = min(sample_per_source, len(candidates))
        sampled.append(rng.choice(candidates, size=n, replace=False))
    rows = np.concatenate(sampled)
    rng.shuffle(rows)
    return rows


def _style_umap(ax: plt.Axes, limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def plot(
    xy: np.ndarray,
    anatomy: np.ndarray,
    membership: np.ndarray,
    counts: dict[str, int],
    out: Path,
    sample_per_source: int,
    seed: int,
    dpi: int,
) -> None:
    if len(xy) != len(anatomy) or len(xy) != len(membership):
        raise ValueError(
            f"row mismatch: UMAP={len(xy)}, anatomy={len(anatomy)}, "
            f"source={len(membership)}"
        )

    unknown = set(np.unique(anatomy)) - set(ANATOMY_GROUPS)
    if unknown:
        raise ValueError(f"unknown anatomical groups: {sorted(unknown)}")

    # Identical coordinate limits are applied explicitly to both views.
    xpad = 0.015 * float(np.ptp(xy[:, 0]))
    ypad = 0.015 * float(np.ptp(xy[:, 1]))
    limits = (
        (float(xy[:, 0].min() - xpad), float(xy[:, 0].max() + xpad)),
        (float(xy[:, 1].min() - ypad), float(xy[:, 1].max() + ypad)),
    )

    fig = plt.figure(figsize=(14.4, 9.0), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 0.32),
        height_ratios=(0.48, 0.52),
        left=0.025,
        right=0.985,
        bottom=0.045,
        top=0.96,
        wspace=0.035,
        hspace=0.06,
    )
    atlas_ax = fig.add_subplot(grid[:, 0])
    legend_ax = fig.add_subplot(grid[0, 1])
    source_ax = fig.add_subplot(grid[1, 1])

    # Main clinical atlas. Draw Other first and large named groups before small
    # ones, keeping rare families visible without changing any coordinates.
    other = anatomy == OTHER
    atlas_ax.scatter(
        xy[other, 0],
        xy[other, 1],
        s=0.55,
        c=ANATOMY_COLORS[OTHER],
        alpha=0.24,
        linewidths=0,
        rasterized=True,
    )
    draw_order = [
        group
        for group in ANATOMY_GROUPS
        if group != OTHER and np.any(anatomy == group)
    ]
    draw_order.sort(key=lambda group: -int((anatomy == group).sum()))
    for group in draw_order:
        mask = anatomy == group
        atlas_ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=0.75,
            c=ANATOMY_COLORS[group],
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
    _style_umap(atlas_ax, limits)
    atlas_ax.set_title(
        "Clinical organization of report-derived concepts",
        loc="left",
        fontsize=16,
        pad=9,
    )
    atlas_ax.text(
        -0.028,
        1.015,
        "a",
        transform=atlas_ax.transAxes,
        fontsize=21,
        fontweight="bold",
        va="bottom",
    )

    legend_ax.axis("off")
    anatomy_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7.5,
            markerfacecolor=ANATOMY_COLORS[group],
            markeredgecolor="none",
            label=group,
        )
        for group in ANATOMY_GROUPS
        if np.any(anatomy == group)
    ]
    legend_ax.legend(
        handles=anatomy_handles,
        loc="center left",
        frameon=False,
        fontsize=10.2,
        labelspacing=0.47,
        handlelength=0.7,
        handletextpad=0.45,
        borderaxespad=0,
    )

    # Same-coordinate dataset control. A single scatter call receives a
    # randomized color vector, preventing either cohort from being drawn as a
    # complete layer on top of the other.
    sampled_rows = balanced_source_sample(membership, sample_per_source, seed)
    sampled_source = membership[sampled_rows]
    source_colors = np.asarray(
        [SOURCE_COLORS[np.uint8(source)] for source in sampled_source]
    )
    source_ax.scatter(
        xy[sampled_rows, 0],
        xy[sampled_rows, 1],
        s=0.65,
        c=source_colors,
        alpha=0.34,
        linewidths=0,
        rasterized=True,
    )
    _style_umap(source_ax, limits)
    source_ax.set_title("Source dataset", fontsize=13, pad=7)
    for spine in source_ax.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.8)

    n_ctr = int((sampled_source == CTR_ONLY).sum())
    n_mer = int((sampled_source == MERLIN_ONLY).sum())
    source_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=SOURCE_COLORS[CTR_ONLY],
            markeredgecolor="none",
            label=f"CT-RATE (n={n_ctr:,})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=SOURCE_COLORS[MERLIN_ONLY],
            markeredgecolor="none",
            label=f"Merlin (n={n_mer:,})",
        ),
    ]
    source_ax.legend(
        handles=source_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        fontsize=9.5,
        ncol=1,
        labelspacing=0.35,
        handlelength=0.7,
        handletextpad=0.4,
    )
    source_ax.text(
        0.5,
        -0.255,
        f"Balanced source-exclusive sample; {counts['Shared exact phrase']:,} "
        "exact shared phrases omitted",
        transform=source_ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.3,
        color="#555555",
        wrap=True,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print(
        "source membership: "
        + ", ".join(f"{name}={value:,}" for name, value in counts.items())
    )
    print(
        f"source inset: {n_ctr:,} CT-RATE-only + {n_mer:,} Merlin-only; "
        f"seed={seed}; identical UMAP coordinates and limits"
    )


def plot_source_only(
    xy: np.ndarray,
    membership: np.ndarray,
    counts: dict[str, int],
    out: Path,
    seed: int,
    dpi: int,
) -> None:
    """Color every concept solely by CT-RATE/Merlin source membership."""
    if len(xy) != len(membership):
        raise ValueError(f"row mismatch: UMAP={len(xy)}, source={len(membership)}")

    # Shuffle all rows before a single scatter call so the larger Merlin bank is
    # not rendered as an opaque layer over CT-RATE. Shared exact phrases remain
    # a third, honest category because assigning them to one corpus is arbitrary.
    rng = np.random.default_rng(seed)
    rows = rng.permutation(len(xy))
    colors = np.asarray(
        [SOURCE_COLORS[np.uint8(source)] for source in membership[rows]]
    )

    xpad = 0.015 * float(np.ptp(xy[:, 0]))
    ypad = 0.015 * float(np.ptp(xy[:, 1]))
    limits = (
        (float(xy[:, 0].min() - xpad), float(xy[:, 0].max() + xpad)),
        (float(xy[:, 1].min() - ypad), float(xy[:, 1].max() + ypad)),
    )

    fig, ax = plt.subplots(figsize=(10.7, 9.0))
    ax.scatter(
        xy[rows, 0],
        xy[rows, 1],
        s=0.65,
        c=colors,
        alpha=0.34,
        linewidths=0,
        rasterized=True,
    )
    # Keep the small shared set visible without changing its coordinates.
    shared = membership == SHARED
    ax.scatter(
        xy[shared, 0],
        xy[shared, 1],
        s=2.0,
        c=SOURCE_COLORS[SHARED],
        alpha=0.85,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    _style_umap(ax, limits)
    ax.set_title("Source dataset", fontsize=16, pad=9)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor=SOURCE_COLORS[source],
            markeredgecolor="none",
            label=f"{label} (n={counts[count_key]:,})",
        )
        for source, label, count_key in (
            (CTR_ONLY, "CT-RATE", "CT-RATE only"),
            (MERLIN_ONLY, "Merlin", "Merlin only"),
            (SHARED, "Both datasets", "Shared exact phrase"),
        )
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=11,
        labelspacing=0.55,
        handlelength=0.8,
        handletextpad=0.45,
    )
    ax.text(
        0.0,
        -0.025,
        "One point per unique observation phrase; all 376,194 concepts shown",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#555555",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print(
        "all concepts: "
        + ", ".join(f"{name}={value:,}" for name, value in counts.items())
        + f"; randomized draw order seed={seed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--umap-dir", type=Path, default=DEFAULT_UMAP_DIR)
    parser.add_argument("--ctrate-jsonl", type=Path, default=DEFAULT_CTRATE)
    parser.add_argument("--merlin-jsonl", type=Path, default=DEFAULT_MERLIN)
    parser.add_argument("--sample-per-source", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="plot all concepts colored only by CT-RATE, Merlin, or both",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.sample_per_source <= 0:
        parser.error("--sample-per-source must be positive")

    with np.load(args.bank, allow_pickle=True) as bank:
        concepts = bank["concepts"]
    xy = np.load(args.umap_dir / "umap2d.npy")
    anatomy = np.load(args.umap_dir / "radlex_anatomy_categories.npy").astype(str)
    membership, counts = source_membership(
        concepts,
        args.ctrate_jsonl,
        args.merlin_jsonl,
    )
    out = args.out or (DEFAULT_SOURCE_ONLY_OUT if args.source_only else DEFAULT_OUT)
    if args.source_only:
        plot_source_only(xy, membership, counts, out, args.seed, args.dpi)
        return
    plot(
        xy,
        anatomy,
        membership,
        counts,
        out,
        args.sample_per_source,
        args.seed,
        args.dpi,
    )


if __name__ == "__main__":
    main()

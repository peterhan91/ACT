from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.decomposition import PCA

from plot_pmbb_combined_atlas import (
    Candidate,
    DEFAULT_CALLOUT_FAMILY_KEYS,
    DEFAULT_DISPLAY_FAMILY_KEYS,
    DEFAULT_PMMB_DIR,
    build_family_callouts,
    concept_first_display_mask,
    concept_report_sets,
    exact_rerank_neighbors,
    recover_reference_pca_transform,
    select_candidates,
    transform_pca,
)
from plot_pmbb_uce_analogue import Family


def test_recover_reference_pca_transform_preserves_saved_frame(tmp_path: Path):
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(240, 12)).astype(np.float32)
    normalized = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    pca = PCA(n_components=5, svd_solver="full")
    saved_scores = pca.fit_transform(normalized).astype(np.float32)

    components, mean, metrics = recover_reference_pca_transform(
        raw,
        saved_scores,
        tmp_path / "transform.npz",
        batch_size=37,
    )
    recovered_scores = transform_pca(raw, components, mean, batch_size=41)

    assert np.sqrt(np.mean((recovered_scores - saved_scores) ** 2)) < 2e-6
    assert metrics["component_orthogonality_max_error"] < 2e-5


def test_exact_rerank_neighbors_sorts_approximate_candidates():
    reference = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        dtype=np.float32,
    )
    query = np.array([[1.1, 0.0], [3.7, 0.0]], dtype=np.float32)
    candidates = np.array([[3, 0, 2, 1], [0, 1, 2, 3]], dtype=np.int64)
    indices, distances = exact_rerank_neighbors(
        query, candidates, reference, k=2, batch_size=1
    )
    assert indices.tolist() == [[1, 2], [3, 2]]
    assert np.allclose(distances, [[0.1, 0.9], [0.3, 1.7]], atol=1e-6)


def _candidate(
    name: str,
    purity: float,
    center: tuple[float, float],
) -> Candidate:
    return Candidate(
        family_index=0,
        family=Family(name, "Finding"),
        reference_points=200,
        pmbb_points=100,
        pmbb_reports=150,
        x0=center[0] - 0.4,
        x1=center[0] + 0.4,
        y0=center[1] - 0.4,
        y1=center[1] + 0.4,
        reference_coverage=0.50,
        pmbb_coverage=0.50,
        strict_all_purity=purity - 0.1,
        retained_label_purity=purity,
        median_center_distance=0.2,
        compactness=0.5,
        pair_leaf_gap=1,
        direct_tree_pair=True,
    )


def test_selection_is_deterministic_and_rejects_spatial_duplicates():
    candidates = [
        _candidate("A", 0.95, (0.0, 0.0)),
        _candidate("B", 0.90, (0.1, 0.1)),
        _candidate("C", 0.85, (3.0, 3.0)),
        _candidate("D", 0.80, (-3.0, -3.0)),
    ]
    selected = select_candidates(
        candidates,
        n_display=3,
        min_reports=100,
        min_points=40,
        min_coverage=0.44,
        min_labelled_purity=0.65,
        max_iou=0.05,
        min_center_separation=2.0,
    )
    assert {value.family.anatomy for value in selected} == {"A", "C", "D"}
    assert sorted(value.display_number for value in selected) == [1, 2, 3]


def test_default_callouts_exclude_only_pleural_but_all_four_families_stay_red():
    pleural_key = "Pleura || Pleural effusion"
    assert len(DEFAULT_DISPLAY_FAMILY_KEYS) == 4
    assert len(DEFAULT_CALLOUT_FAMILY_KEYS) == 3
    assert pleural_key in DEFAULT_DISPLAY_FAMILY_KEYS
    assert pleural_key not in DEFAULT_CALLOUT_FAMILY_KEYS
    assert DEFAULT_CALLOUT_FAMILY_KEYS == tuple(
        key for key in DEFAULT_DISPLAY_FAMILY_KEYS if key != pleural_key
    )

    keys = np.asarray(
        list(DEFAULT_DISPLAY_FAMILY_KEYS)
        + [DEFAULT_DISPLAY_FAMILY_KEYS[0], "Other || Other"],
        dtype=object,
    )
    strict = np.asarray([True, True, True, True, False, True])

    displayed = concept_first_display_mask(keys, strict)
    expected = strict & np.isin(
        keys, np.asarray(DEFAULT_DISPLAY_FAMILY_KEYS, dtype=object)
    )

    assert displayed.dtype == np.bool_
    assert np.array_equal(displayed, expected)
    assert displayed.tolist() == [True, True, True, True, False, False]


def test_full_bank_default_and_missing_report_trace_are_explicit(tmp_path: Path):
    assert DEFAULT_PMMB_DIR.name == "pmbb_qwen36_extracted16896_f2llm_20260715"
    assert DEFAULT_PMMB_DIR.parent.name == "banks"
    include = np.asarray([True, False, True], dtype=bool)
    assert concept_report_sets(tmp_path / "missing.jsonl", 3, include) is None


def _tree_context(centroids: np.ndarray):
    tree = linkage(np.asarray(centroids, dtype=float), method="complete")
    _, nodes = to_tree(tree, rd=True)
    parent = {}
    for node in nodes:
        if node.left is not None:
            parent[node.left.id] = node
        if node.right is not None:
            parent[node.right.id] = node
    return nodes, parent


def test_broad_family_callout_uses_exact_35_percent_core_and_six_percent_floor():
    family = Family("A", "Finding A")
    # Fourteen central rows and 26 distant rows. With n=40, 35% is exactly 14.
    central_x = np.r_[-np.arange(1.0, 8.0), np.arange(1.0, 8.0)]
    outer_x = np.r_[-np.arange(20.0, 33.0), np.arange(20.0, 33.0)]
    pmbb_xy = np.column_stack(
        (np.concatenate((central_x, outer_x)), np.zeros(40))
    ).astype(np.float32)
    atlas_xy = np.asarray([[-50.0, -25.0], [50.0, 25.0]], dtype=np.float32)
    nodes, parent = _tree_context(np.asarray([[0.0, 0.0], [0.01, 0.0]]))

    callouts = build_family_callouts(
        atlas_xy=atlas_xy,
        pmbb_xy=pmbb_xy,
        reference_keys=np.asarray([family.key, family.key], dtype=object),
        pmbb_keys=np.asarray([family.key] * len(pmbb_xy), dtype=object),
        reference_nonoverlap=np.ones(len(atlas_xy), dtype=bool),
        pmbb_nonoverlap=np.ones(len(pmbb_xy), dtype=bool),
        families=[family],
        report_sets=[{str(index)} for index in range(len(pmbb_xy))],
        tree_nodes=nodes,
        parent=parent,
        callout_family_keys=(family.key,),
        display_family_keys=(family.key,),
        central_fraction=0.35,
        min_side_span_fraction=0.06,
    )

    assert len(callouts) == 1
    value = callouts[0]
    assert value.family.key == family.key
    assert value.core_points == 14
    assert value.family_target_points == 40
    assert value.pmbb_target_points == 14
    # The core x range is already wider than 6% of the 100-unit atlas span.
    assert np.allclose([value.x0, value.x1], [-7.0, 7.0], atol=1e-12)
    # All y values and the y-MAD are zero, so the box must remain finite and
    # expand symmetrically to 6% of the 50-unit atlas span.
    assert np.allclose([value.y0, value.y1], [-1.5, 1.5], atol=1e-12)
    assert np.isfinite([value.x0, value.x1, value.y0, value.y1]).all()


def test_broad_callouts_are_permutation_invariant_ordered_and_display_only():
    rng = np.random.default_rng(12)
    families = [
        Family(*key.split(" || ", maxsplit=1)) for key in DEFAULT_DISPLAY_FAMILY_KEYS
    ]
    offsets = np.asarray([-30.0, -10.0, 10.0, 30.0])
    central = np.r_[-np.arange(0.1, 0.8, 0.1), np.arange(0.1, 0.8, 0.1)]
    outer = np.r_[-np.arange(2.0, 15.0), np.arange(2.0, 15.0)]
    local_x = np.concatenate((central, outer))
    pmbb_xy = np.vstack(
        [
            np.column_stack(
                (
                    offset + local_x,
                    rng.normal(0.0, 0.03, size=len(local_x)),
                )
            )
            for offset in offsets
        ]
    ).astype(np.float32)
    pmbb_keys = np.concatenate(
        [np.asarray([family.key] * len(local_x), dtype=object) for family in families]
    )
    atlas_xy = np.vstack(
        (
            np.column_stack((np.repeat(offsets, 2), np.tile([-20.0, 20.0], 4))),
            np.asarray([[-50.0, -20.0], [50.0, 20.0]]),
        )
    ).astype(np.float32)
    reference_keys = np.asarray(
        [family.key for family in families for _ in range(2)]
        + [families[0].key, families[-1].key],
        dtype=object,
    )
    centroid_xy = np.vstack(
        (
            np.column_stack((offsets, np.zeros(len(offsets)))),
            np.column_stack((offsets + 0.01, np.zeros(len(offsets)))),
        )
    )
    nodes, parent = _tree_context(centroid_xy)

    kwargs = dict(
        atlas_xy=atlas_xy,
        pmbb_xy=pmbb_xy,
        reference_keys=reference_keys,
        reference_nonoverlap=np.ones(len(atlas_xy), dtype=bool),
        pmbb_nonoverlap=np.ones(len(pmbb_xy), dtype=bool),
        families=families,
        report_sets=[{str(index)} for index in range(len(pmbb_xy))],
        tree_nodes=nodes,
        parent=parent,
        callout_family_keys=DEFAULT_CALLOUT_FAMILY_KEYS,
        display_family_keys=DEFAULT_DISPLAY_FAMILY_KEYS,
        central_fraction=0.35,
        min_side_span_fraction=0.06,
    )
    original = build_family_callouts(pmbb_keys=pmbb_keys, **kwargs)

    permutation = rng.permutation(len(pmbb_xy))
    permuted_kwargs = dict(kwargs)
    permuted_kwargs["pmbb_xy"] = pmbb_xy[permutation]
    permuted_kwargs["pmbb_nonoverlap"] = np.ones(len(pmbb_xy), dtype=bool)
    permuted_kwargs["report_sets"] = [kwargs["report_sets"][i] for i in permutation]
    permuted = build_family_callouts(
        pmbb_keys=pmbb_keys[permutation],
        **permuted_kwargs,
    )

    def geometry(values):
        return [
            (
                value.family.key,
                value.x0,
                value.x1,
                value.y0,
                value.y1,
                value.core_points,
                value.pmbb_target_points,
                value.family_target_points,
            )
            for value in values
        ]

    original_geometry = geometry(original)
    permuted_geometry = geometry(permuted)
    assert [value[0] for value in original_geometry] == list(
        DEFAULT_CALLOUT_FAMILY_KEYS
    )
    assert [value[0] for value in permuted_geometry] == list(
        DEFAULT_CALLOUT_FAMILY_KEYS
    )
    assert np.allclose(
        np.asarray([value[1:] for value in original_geometry], dtype=float),
        np.asarray([value[1:] for value in permuted_geometry], dtype=float),
        atol=1e-12,
    )

    display_before = concept_first_display_mask(
        pmbb_keys,
        kwargs["pmbb_nonoverlap"],
        family_keys=DEFAULT_DISPLAY_FAMILY_KEYS,
    )
    inside_selected_box = np.zeros(len(pmbb_xy), dtype=bool)
    for value in original:
        inside_selected_box |= (
            (pmbb_xy[:, 0] >= value.x0)
            & (pmbb_xy[:, 0] <= value.x1)
            & (pmbb_xy[:, 1] >= value.y0)
            & (pmbb_xy[:, 1] <= value.y1)
        )
    display_after = concept_first_display_mask(
        pmbb_keys,
        kwargs["pmbb_nonoverlap"],
        family_keys=DEFAULT_DISPLAY_FAMILY_KEYS,
    )
    assert np.array_equal(display_before, display_after)
    assert display_after.sum() == len(pmbb_xy)
    assert np.any(display_after & ~inside_selected_box)
    pleural = pmbb_keys == "Pleura || Pleural effusion"
    assert display_after[pleural].all()
    assert all(value.family.key != "Pleura || Pleural effusion" for value in original)

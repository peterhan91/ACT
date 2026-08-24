import json

import numpy as np

from plot_pmbb_uce_analogue import (
    Family,
    alignment_table,
    centroid_matrix,
    pmbb_cohort_bits,
    sample_reference_rows,
    scanpy_style_linkage,
)


def test_pmbb_cohort_bits_tracks_cross_cohort_concepts(tmp_path):
    path = tmp_path / "occurrences.jsonl"
    records = [
        {"concept_index": 0, "cohort": "pmbb_chest_nc"},
        {"concept_index": 1, "cohort": "pmbb_abd_ce"},
        {"concept_index": 2, "cohort": "pmbb_chest_nc"},
        {"concept_index": 2, "cohort": "pmbb_abd_ce"},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert pmbb_cohort_bits(path, 3).tolist() == [1, 2, 3]


def test_reference_sample_keeps_one_shared_copy():
    scope = np.ones(10, dtype=bool)
    nonoverlap = np.array([True] * 7 + [False] * 3)
    sampled, shared, audit = sample_reference_rows(scope, nonoverlap, 4, seed=0)
    assert len(sampled) == 4
    assert set(sampled).issubset(set(range(7)))
    assert shared.tolist() == [7, 8, 9]
    assert audit["total_reference_coordinates"] == 7


def test_centroids_and_scanpy_style_tree_are_finite():
    values = np.array(
        [
            [1.0, 0.0, 0.1],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.1],
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 1.0],
            [0.1, 0.0, 0.9],
        ],
        dtype=np.float32,
    )
    centroids, counts = centroid_matrix(values, np.repeat(np.arange(3), 2), 3)
    tree, distance = scanpy_style_linkage(centroids)
    assert counts.tolist() == [2, 2, 2]
    assert tree.shape == (2, 4)
    assert np.isfinite(distance).all()
    assert np.allclose(np.diag(distance), 0.0)


def test_alignment_table_is_a_taxonomy_consistency_check():
    reference = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    new = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    families = [Family("A", "X"), Family("B", "X")]
    rows, summary = alignment_table(
        reference,
        new,
        families,
        seed=0,
        permutation_replicates=100,
    )
    assert len(rows) == 2
    assert summary["top1_accuracy_reference_candidates_only"] == 1.0
    assert "not independent clinical validation" in summary["claim_boundary"]

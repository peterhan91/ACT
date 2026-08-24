from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

from plot_radlex_embedding_distance import (
    BRANCH_BASE_COLORS,
    DEFAULT_RADLEX,
    DISTRIBUTION_LINE_COLOR,
    FINAL_FIGURE_STEM,
    LEGACY_FIGURE_STEMS,
    PLOT_BRANCH_ORDER,
    UCE_BRANCH_TITLES,
    _branch_hop_colors,
    _draw_embedding_distance_distribution,
    _format_rotated_pvalue,
    branch_term_index,
    map_concepts,
    normalized,
    parse_radlex,
    plot_uce_two_branch,
    remove_legacy_figure_files,
)


def test_uce_branch_titles() -> None:
    assert UCE_BRANCH_TITLES == {
        "RID5": "Imaging observations",
        "RID34785": "Clinical finding",
    }


def test_branch_palettes_use_requested_hues_and_darkening() -> None:
    assert BRANCH_BASE_COLORS == {
        "RID34785": "#2F67B1",
        "RID5": "#7651A5",
    }
    for root_rid in BRANCH_BASE_COLORS:
        colors = _branch_hop_colors(root_rid, max_hop=5)
        assert list(colors) == [1, 2, 3, 4, 5]
        assert len({tuple(color) for color in colors.values()}) == 5
        assert sum(colors[1]) > sum(colors[5])
    assert DISTRIBUTION_LINE_COLOR == "#000000"


def test_distribution_mean_marker_is_dashed_and_at_raw_mean() -> None:
    values = np.asarray([0.72, 0.91, 1.03, 1.17, 1.31], dtype=float)
    fig, ax = plt.subplots()
    _draw_embedding_distance_distribution(
        ax,
        values,
        fill_color=(0.65, 0.75, 0.9),
        line_color=DISTRIBUTION_LINE_COLOR,
        alpha=0.86,
        clip=(0.0, 1.5),
    )

    mean_markers = [
        collection
        for collection in ax.collections
        if isinstance(collection, LineCollection)
    ]
    assert len(mean_markers) == 1
    segment = mean_markers[0].get_segments()[0]
    assert segment[:, 0] == pytest.approx([values.mean(), values.mean()])
    assert segment[0, 1] == pytest.approx(0.0)
    assert segment[1, 1] > 0.0
    assert len(mean_markers[0].get_linestyle()[0][1]) > 0
    assert mean_markers[0].get_colors()[0, :3] == pytest.approx([0.0, 0.0, 0.0])
    filled_ridges = [
        collection
        for collection in ax.collections
        if isinstance(collection, PolyCollection)
    ]
    assert len(filled_ridges) == 1
    assert filled_ridges[0].get_edgecolors()[0, :3] == pytest.approx([0.0, 0.0, 0.0])
    plt.close(fig)


def test_rotated_pvalue_format_uses_numeric_values() -> None:
    assert _format_rotated_pvalue(0.05938412701232177) == "p = 0.05938"
    assert _format_rotated_pvalue(5.0383451679863074e-23) == "p = 5.038e-23"
    assert _format_rotated_pvalue(float("nan")) == "p = NA"


def test_final_two_branch_layout_matches_reference(tmp_path, monkeypatch) -> None:
    pair_tables = {}
    statistics = {}
    for branch_offset, root_rid in enumerate(PLOT_BRANCH_ORDER):
        records = []
        for hop in range(1, 6):
            for value in np.linspace(0.75, 1.35, 5):
                records.append(
                    {
                        "ontology_distance": hop,
                        "embedding_distance": value + branch_offset * 0.01,
                    }
                )
        pair_tables[root_rid] = pd.DataFrame.from_records(records)
        statistics[root_rid] = {
            "overall": {
                "adjacent_tests": [
                    {"lower_hop": hop, "rid_pair_welch_p_holm": 0.01 * hop}
                    for hop in range(1, 5)
                ]
            }
        }

    captured = {}

    def capture_figure(figure, *_args, **_kwargs) -> None:
        captured["figure"] = figure

    monkeypatch.setattr(plt.Figure, "savefig", capture_figure)
    monkeypatch.setattr(plt, "close", lambda _figure: None)
    plot_uce_two_branch(
        pair_tables,
        statistics,
        output_stem=tmp_path / FINAL_FIGURE_STEM,
        max_hop=5,
        dpi=72,
    )

    figure = captured["figure"]
    ridge_axes = [axis for axis in figure.axes if axis.axison]
    assert len(ridge_axes) == 10
    assert [axis.get_title() for axis in ridge_axes if axis.get_title()] == [
        "Clinical finding",
        "Imaging observations",
    ]
    assert sum(axis.get_xlabel() == "Embedding distance in F2LLM" for axis in ridge_axes) == 2
    assert all(axis.get_xlim() == pytest.approx((0.0, 1.5)) for axis in ridge_axes)
    grid = ridge_axes[0].get_subplotspec().get_gridspec()
    assert grid.get_geometry() == (5, 4)
    assert grid.get_width_ratios() == pytest.approx([5.4, 0.70, 5.4, 0.70])
    assert (grid.left, grid.right, grid.bottom, grid.top) == pytest.approx(
        (0.07, 0.985, 0.09, 0.94)
    )
    assert (grid.hspace, grid.wspace) == pytest.approx((0.08, 0.06))
    plt.close(figure)


def test_cleanup_removes_only_superseded_figures(tmp_path) -> None:
    legacy_paths = []
    for stem in LEGACY_FIGURE_STEMS:
        for suffix in (".png", ".pdf"):
            path = tmp_path / f"{stem}{suffix}"
            path.write_bytes(b"legacy")
            legacy_paths.append(path)

    retained = [
        tmp_path / f"{FINAL_FIGURE_STEM}.png",
        tmp_path / f"{FINAL_FIGURE_STEM}.pdf",
        tmp_path / "statistics.json",
        tmp_path / "distance_summary_by_hop.csv",
    ]
    for path in retained:
        path.write_bytes(b"keep")

    removed = remove_legacy_figure_files(tmp_path)
    assert set(removed) == set(legacy_paths)
    assert all(not path.exists() for path in legacy_paths)
    assert all(path.is_file() for path in retained)


def test_split_rdf_description_annotations_are_merged(tmp_path) -> None:
    owl_path = tmp_path / "split_annotations.owl"
    owl_path.write_text(
        """<?xml version="1.0"?>
<rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:owl="http://www.w3.org/2002/07/owl#"
    xmlns:RID="http://www.radlex.org/RID/">
  <rdf:Description rdf:about="http://www.radlex.org/RID/RIDCHILD">
    <rdfs:label xml:lang="en">description label</rdfs:label>
    <rdfs:label xml:lang="de">deutsche Bezeichnung</rdfs:label>
    <RID:Synonym>child alias</RID:Synonym>
    <RID:Acronym xml:lang="en">CA</RID:Acronym>
    <RID:Preferred_Name_for_Obsolete>RIDREPLACEMENT</RID:Preferred_Name_for_Obsolete>
  </rdf:Description>
  <owl:Class rdf:about="http://www.radlex.org/RID/RIDROOT">
    <rdfs:label xml:lang="en">root label</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://www.radlex.org/RID/RIDCHILD">
    <rdfs:subClassOf rdf:resource="http://www.radlex.org/RID/RIDROOT"/>
    <rdfs:label xml:lang="en">class preferred label</rdfs:label>
  </owl:Class>
</rdf:RDF>
"""
    )

    parsed = parse_radlex(owl_path)
    assert parsed.hierarchy.has_edge("RIDROOT", "RIDCHILD")
    assert parsed.preferred_labels["RIDCHILD"] == "class preferred label"
    assert parsed.terms["RIDCHILD"] == {
        "class preferred label",
        "description label",
        "child alias",
        "CA",
    }
    assert "deutsche Bezeichnung" not in parsed.terms["RIDCHILD"]
    assert "RIDCHILD" in parsed.obsolete


@pytest.fixture(scope="module")
def radlex_43():
    if not DEFAULT_RADLEX.exists():
        pytest.skip("the pinned RadLex 4.3 OWL file is not available")
    return parse_radlex(DEFAULT_RADLEX)


def test_description_annotations_are_merged(radlex_43) -> None:
    recovered = {
        "RID28493": "atelectasis",
        "RID34539": "pleural effusion",
    }
    for rid, expected in recovered.items():
        assert radlex_43.preferred_labels[rid] == expected
        assert expected in {normalized(term) for term in radlex_43.terms[rid]}
    assert radlex_43.hierarchy.has_edge("RID28506", "RID28493")
    assert "RID6644" in radlex_43.obsolete


def test_every_active_target_branch_node_has_a_term(radlex_43) -> None:
    expected_active_nodes = {
        "RID5": 1_213,
        "RID34785": 2_239,
    }
    for root_rid, expected in expected_active_nodes.items():
        _, _, _, audit = branch_term_index(radlex_43, root_rid)
        assert audit["active_nonroot_nodes"] == expected
        assert audit["active_term_nodes"] == expected
        assert audit["active_nodes_without_terms"] == 0


def test_recovered_terms_follow_conservative_mapping_rules(radlex_43) -> None:
    concepts = np.asarray(
        [
            "simulated pleural effusion measuring 12 mm in thickness",
            "atelectasis",
            "dependent atelectasis present",
        ],
        dtype=object,
    )
    mapping, outcomes, _, _ = map_concepts(concepts, radlex_43, "RID34785")
    by_index = mapping.set_index("concept_index")

    assert by_index.loc[0, "rid"] == "RID34539"
    assert by_index.loc[1, "rid"] == "RID28493"
    assert 2 not in by_index.index
    assert outcomes["accepted"] == 2
    assert outcomes["unsafe_single_token_in_clause"] == 1

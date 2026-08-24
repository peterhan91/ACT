#!/usr/bin/env python3
"""Build a transparent clinical audit of the natural f2llm top-10 concepts.

The coefficient source is the Adam (lr=3e-2) INSPECT-221 linear probe used by
the shortcut/concept-coefficient figure:

  outputs/v1/external/
    phenotype__linear_f2llm__seed{1..20}__lr3e-2__coefs.pt

The natural ranking is recomputed directly from the current seed bundles and
all 376,194 raw f2llm concept embeddings, using the same coefficient convention
as ``compute_shortcut_coefs.py``.  This script does not silently replace a
rejected concept with a lower-ranked one.  It audits each natural top-10
concept and emits both:

  * natural_top10: all ten concepts, each with the audit decision/reason; and
  * audited_top10: the retained subset, in the original natural rank order.

The output therefore distinguishes a naturally clean explanation from a
post-hoc direct-tier reranking/intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

DEFAULT_OUTPUT = RESULTS / "f2llm_20seed_natural_top10_clinical_audit.json"
DEFAULT_EMBEDDINGS = ROOT / "_emb_f2llm_tmp.npy"
DEFAULT_BANK = ROOT / "concept_bank.f2llm_emb.npz"
DEFAULT_ANNOTATIONS = ROOT / "concept_annotations.json"
SEED_TEMPLATE = (
    ROOT
    / "outputs/v1/external/"
    "phenotype__linear_f2llm__seed{seed}__lr3e-2__coefs.pt"
)

N_SEEDS = 20
N_TOP = 10
N_CONCEPTS_EXPECTED = 376_194
T_CRIT_DF19 = 2.093024054408263  # two-sided 95% CI for the across-seed mean


# A direct imaging pattern can be real while still failing to identify an
# etiologic phenotype.  These target-level policies prevent that overclaim.
TARGET_BLOCKS = {
    "Bacterial pneumonia": (
        "etiologic_target_not_ct_definitional",
        "Air-space disease can support pneumonia but CTPA cannot establish a bacterial cause.",
    ),
    "Pneumococcal pneumonia": (
        "organism_target_not_ct_definitional",
        "CT appearance cannot establish Streptococcus pneumoniae as the organism.",
    ),
    "Pneumonitis due to inhalation of food or vomitus": (
        "causal_target_not_ct_definitional",
        "Air-space disease is nonspecific and does not establish aspiration as the cause.",
    ),
}


# CTPA routinely covers the chest and only a variable portion of the upper
# abdomen.  These broad abdominopelvic targets are not eligible for a strict
# CTPA-grounded panel even when a report phrase is anatomically plausible.
CTPA_FOV_BLOCKS = {
    "Other disorders of peritoneum",
    "Ascites (non malignant)",
    "Paralytic ileus",
    "Other disorders of intestine",
}


BRANCH_SPECIFIC = {
    "Pleurisy; pleural effusion": (
        "Retained concepts support the pleural-effusion branch, not pleuritic symptoms."
    ),
    "Pulmonary collapse; interstitial and compensatory emphysema": (
        "Retained concepts may support only the atelectasis/collapse or emphysema branch of this composite label."
    ),
}


IMAGING_SUPPORT_ONLY = {
    "Pneumonia": (
        "Retained air-space findings are imaging support for pneumonia, not a microbiologic diagnosis."
    ),
}


SOLITARY_MISMATCH = re.compile(
    r"\bnodules\b|"
    r"\b(multiple|bilateral|several|innumerable|numerous|diffuse|scattered|few)\b"
    r".*\b(nodules?|masses?)\b|\b(nodules?|masses?)\b.*"
    r"\b(multiple|bilateral|several|innumerable|numerous|diffuse|scattered|few)\b",
    re.IGNORECASE,
)

SOLITARY_SINGLE_DIRECT = re.compile(
    r"\bnodule\b.*\b(right|left) (upper|middle|lower) lobe\b",
    re.IGNORECASE,
)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_audit_tools():
    # Keep this import local so the script remains explicit about the repo-local
    # deterministic clinical profile used for screening.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from trusted_concept_space import (  # pylint: disable=import-outside-toplevel
        GLOBAL_EXCLUDE,
        auto_profile,
        classify_concept,
        compile_all,
        matched_patterns,
    )

    return GLOBAL_EXCLUDE, auto_profile, classify_concept, compile_all, matched_patterns


def load_seed_data() -> tuple[np.ndarray, list[str], list[str], dict[str, np.ndarray], list[Path]]:
    """Return W[S,L,D], label order, phecodes, per-label AUROCs, and paths."""
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
        labels = [str(x) for x in bundle["labels"]]
        phecodes = [str(x) for x in bundle["phecodes"]]
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
        if not math.isclose(float(bundle["lr"]), 0.03, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"expected Adam lr=3e-2: {path}")
        weights.append(bundle["W"].numpy().astype(np.float32, copy=False))
        for label in labels:
            aucs[label].append(float(bundle["test_per_label_auc"][label]))
        paths.append(path)

    assert labels_ref is not None and phecodes_ref is not None
    return (
        np.stack(weights, axis=0),
        labels_ref,
        phecodes_ref,
        {label: np.asarray(values, dtype=np.float64) for label, values in aucs.items()},
        paths,
    )


def bank_mean(embeddings: np.memmap, chunk_size: int = 20_000) -> np.ndarray:
    total = np.zeros(embeddings.shape[1], dtype=np.float64)
    for start in range(0, embeddings.shape[0], chunk_size):
        total += embeddings[start : start + chunk_size].astype(np.float64).sum(axis=0)
    return (total / embeddings.shape[0]).astype(np.float32)


def centered_normalized(rows: np.ndarray, mean: np.ndarray) -> np.ndarray:
    out = rows.astype(np.float32) - mean
    out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-9)
    return out


def full_bank_top10(
    weights: np.ndarray,
    embedding_path: Path,
    *,
    chunk_size: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Stream exact raw-coefficient top-10 for every phenotype.

    Returns indices[L,K], scores[L,K], bank mean, and embedding shape.  Ranking
    uses mean(W_seed) dot normalize(raw_embedding - bank_mean), which is the
    coefficient convention used by the target bar/scatter figure.
    """
    emb = np.load(embedding_path, mmap_mode="r")
    if emb.ndim != 2 or emb.shape[0] != N_CONCEPTS_EXPECTED:
        raise ValueError(f"unexpected f2llm embedding shape: {emb.shape}")
    if weights.shape[2] != emb.shape[1]:
        raise ValueError(f"probe/embedding dimension mismatch: {weights.shape} vs {emb.shape}")

    mean = bank_mean(emb)
    mean_weight = weights.mean(axis=0, dtype=np.float32)
    n_labels = mean_weight.shape[0]
    best_scores = np.full((n_labels, N_TOP), -np.inf, dtype=np.float32)
    best_indices = np.full((n_labels, N_TOP), -1, dtype=np.int64)

    for start in range(0, emb.shape[0], chunk_size):
        stop = min(start + chunk_size, emb.shape[0])
        concept_rows = centered_normalized(emb[start:stop], mean)
        scores = mean_weight @ concept_rows.T

        local_k = min(N_TOP, scores.shape[1])
        local_columns = np.argpartition(scores, scores.shape[1] - local_k, axis=1)[:, -local_k:]
        local_scores = np.take_along_axis(scores, local_columns, axis=1)
        local_indices = local_columns.astype(np.int64) + start

        merged_scores = np.concatenate([best_scores, local_scores], axis=1)
        merged_indices = np.concatenate([best_indices, local_indices], axis=1)
        keep_columns = np.argpartition(
            merged_scores, merged_scores.shape[1] - N_TOP, axis=1
        )[:, -N_TOP:]
        best_scores = np.take_along_axis(merged_scores, keep_columns, axis=1)
        best_indices = np.take_along_axis(merged_indices, keep_columns, axis=1)

    # Descending coefficient with concept index as the deterministic tie-break.
    for label_idx in range(n_labels):
        order = sorted(
            range(N_TOP),
            key=lambda j: (-float(best_scores[label_idx, j]), int(best_indices[label_idx, j])),
        )
        best_scores[label_idx] = best_scores[label_idx, order]
        best_indices[label_idx] = best_indices[label_idx, order]

    if np.any(best_indices < 0) or not np.all(best_scores[:, :-1] >= best_scores[:, 1:]):
        raise RuntimeError("invalid streamed full-bank top-10 result")
    return best_indices, best_scores, mean, tuple(emb.shape)


def selected_embeddings_by_index(
    embedding_path: Path,
    indices: np.ndarray,
    mean: np.ndarray,
) -> dict[int, np.ndarray]:
    emb = np.load(embedding_path, mmap_mode="r")
    unique = sorted({int(x) for x in indices.ravel()})
    rows = centered_normalized(emb[unique], mean)
    return {index: rows[row] for row, index in enumerate(unique)}


def exclusion_reason(
    global_hits: list[str],
    label_hits: list[str],
    global_patterns: list[str],
) -> tuple[str, str]:
    positions = {pattern: idx for idx, pattern in enumerate(global_patterns)}
    hit_positions = {positions[h] for h in global_hits if h in positions}
    if hit_positions.intersection({0, 7, 8}):
        return "negated_or_absent", "The phrase negates or removes the target finding."
    if 1 in hit_positions:
        return "normal_finding", "A normal/unremarkable phrase is not positive target evidence."
    if 2 in hit_positions:
        return "technical_limitation", "A technical limitation is not positive target evidence."
    if 4 in hit_positions:
        return "procedure_or_device", "Procedure/device/postoperative wording is not direct phenotype evidence."
    if 5 in hit_positions:
        return "uncertain_language", "Uncertain language is excluded from the strict direct tier."
    if 3 in hit_positions:
        return "temporal_comparison", "The rank is driven by temporal/comparison wording rather than the finding alone."
    if 6 in hit_positions:
        return "history_or_treatment", "History/treatment wording is not direct current imaging evidence."
    if hit_positions.intersection({9, 10}):
        return "nonpulmonary_emphysema", "The phrase refers to nonpulmonary soft-tissue or visceral gas."
    if label_hits:
        return "label_specific_exclusion", "The phrase matches a label-specific confound/exclusion rule."
    return "excluded_by_audit_rule", "The phrase matches a conservative audit exclusion."


def audit_concept(
    *,
    label: str,
    concept: str,
    profile,
    global_exclude,
    direct_patterns,
    associated_patterns,
    label_exclude,
    classify_concept,
    matched_patterns,
    global_pattern_strings: list[str],
) -> dict:
    tier, score, matches, exclusion_matches = classify_concept(
        concept,
        profile,
        global_exclude,
        direct_patterns,
        associated_patterns,
        label_exclude,
    )
    global_hits = matched_patterns(concept, global_exclude)
    label_hits = matched_patterns(concept, label_exclude)
    manual_override = None

    # The bank often writes a lobe location without repeating "lung"; the
    # generic profile otherwise misses a literal singular pulmonary nodule.
    if (
        tier == "rejected"
        and label == "Solitary pulmonary nodule"
        and not global_hits
        and not label_hits
        and SOLITARY_SINGLE_DIRECT.search(concept)
    ):
        tier = "direct"
        score = 100 - min(len(concept.split()), 30)
        matches = ["manual: singular nodule with explicit lung-lobe location"]
        manual_override = "solitary_nodule_lobe_location"

    decision = "reject"
    reason_code = "target_mismatch"
    reason = "The phrase does not match a reviewed direct or associated rule for this phenotype."
    evidence_scope = "none"
    retain = False

    if tier == "excluded":
        reason_code, reason = exclusion_reason(global_hits, label_hits, global_pattern_strings)
    elif tier == "associated":
        reason_code = "association_only"
        reason = "The finding is associated with, but not direct evidence for, this phenotype."
        evidence_scope = "associated"
    elif tier == "direct":
        retain = True
        decision = "retain_direct"
        reason_code = "direct_ct_finding"
        reason = "The phrase is an affirmative, target-matched finding visible within the CTPA field."
        evidence_scope = "direct"

        if label in TARGET_BLOCKS:
            retain = False
            decision = "reject"
            reason_code, reason = TARGET_BLOCKS[label]
            evidence_scope = "imaging_pattern_not_target_specific"
        elif label in CTPA_FOV_BLOCKS:
            retain = False
            decision = "reject"
            reason_code = "ctpa_field_of_view_mismatch"
            reason = "This broad abdominopelvic target is not reliably assessed over the routine CTPA field of view."
            evidence_scope = "outside_or_incomplete_ctpa_fov"
        elif label == "Solitary pulmonary nodule" and SOLITARY_MISMATCH.search(concept):
            retain = False
            decision = "reject"
            reason_code = "solitary_multiplicity_mismatch"
            reason = "Plural/multiple nodules do not directly support a solitary-nodule target."
            evidence_scope = "target_mismatch"
        elif label in BRANCH_SPECIFIC:
            decision = "retain_branch_specific"
            reason_code = "direct_composite_branch"
            reason = BRANCH_SPECIFIC[label]
            evidence_scope = "direct_for_one_label_branch"
        elif label in IMAGING_SUPPORT_ONLY:
            decision = "retain_imaging_support"
            reason_code = "direct_imaging_pattern_not_etiology"
            reason = IMAGING_SUPPORT_ONLY[label]
            evidence_scope = "imaging_supportive_not_specific"

    return {
        "classifier_tier": tier,
        "classifier_score": int(score),
        "decision": decision,
        "retain": bool(retain),
        "reason_code": reason_code,
        "reason": reason,
        "evidence_scope": evidence_scope,
        "matched_patterns": matches,
        "exclusion_patterns": exclusion_matches,
        "manual_override": manual_override,
    }


def title_for(label: str, mean: float, lo: float, hi: float) -> str:
    # The AUROC intentionally begins on line 2, matching the requested figure style.
    return f"{label}\n(AUROC {mean:.3f}, 95% CI {lo:.3f}-{hi:.3f})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--concept-bank", type=Path, default=DEFAULT_BANK)
    args = parser.parse_args()

    (
        GLOBAL_EXCLUDE,
        auto_profile,
        classify_concept,
        compile_all,
        matched_patterns,
    ) = _load_audit_tools()
    global_exclude = compile_all(GLOBAL_EXCLUDE)

    weights, labels, phecodes, auc_by_label, seed_paths = load_seed_data()
    top_indices, top_scores, mean_vector, embedding_shape = full_bank_top10(
        weights, args.embeddings
    )
    concept_bank = np.load(args.concept_bank, allow_pickle=True)
    concepts = concept_bank["concepts"]
    bank_size = int(len(concepts))
    if bank_size != N_CONCEPTS_EXPECTED or embedding_shape[0] != bank_size:
        raise ValueError("embedding/concept-bank row-count mismatch")
    embedding_by_index = selected_embeddings_by_index(
        args.embeddings, top_indices, mean_vector
    )

    with DEFAULT_ANNOTATIONS.open() as f:
        annotation_data = json.load(f).get("inspect", {})
    categories = {
        label: str(entry.get("category", ""))
        for label, entry in annotation_data.items()
        if isinstance(entry, dict)
    }

    entries: list[dict] = []
    global_decisions: Counter[str] = Counter()
    global_tiers: Counter[str] = Counter()

    for label_idx, (label, phecode) in enumerate(zip(labels, phecodes)):
        aucs = auc_by_label[label]
        auc_mean = float(aucs.mean())
        auc_std = float(aucs.std(ddof=1))
        half_width = T_CRIT_DF19 * auc_std / math.sqrt(N_SEEDS)
        auc_lo, auc_hi = auc_mean - half_width, auc_mean + half_width

        profile = auto_profile(label, categories.get(label, ""))
        direct_patterns = compile_all(profile.direct)
        associated_patterns = compile_all(profile.associated)
        label_exclude = compile_all(profile.exclude)

        natural_rows: list[dict] = []
        retained_rows: list[dict] = []
        decision_counts: Counter[str] = Counter()
        tier_counts: Counter[str] = Counter()

        for rank, (concept_index, selected_score) in enumerate(
            zip(top_indices[label_idx], top_scores[label_idx]), start=1
        ):
            concept_index = int(concept_index)
            concept = str(concepts[concept_index])
            vector = embedding_by_index[concept_index]
            per_seed = weights[:, label_idx, :] @ vector
            if not math.isclose(
                float(per_seed.mean()), float(selected_score), rel_tol=0.0, abs_tol=5e-5
            ):
                raise ValueError(f"mean coefficient mismatch for {label}, rank {rank}")
            audit = audit_concept(
                label=label,
                concept=concept,
                profile=profile,
                global_exclude=global_exclude,
                direct_patterns=direct_patterns,
                associated_patterns=associated_patterns,
                label_exclude=label_exclude,
                classify_concept=classify_concept,
                matched_patterns=matched_patterns,
                global_pattern_strings=list(GLOBAL_EXCLUDE),
            )
            tier_counts[audit["classifier_tier"]] += 1
            decision_counts[audit["reason_code"]] += 1
            global_tiers[audit["classifier_tier"]] += 1
            global_decisions[audit["reason_code"]] += 1

            item = {
                "natural_rank": rank,
                "concept_index": concept_index,
                "concept": concept,
                "coefficient": {
                    "mean": float(per_seed.mean()),
                    "std": float(per_seed.std(ddof=1)),
                    "per_seed": [float(x) for x in per_seed],
                },
                "coefficient_of_mean_weight": float(selected_score),
                "audit": audit,
            }
            natural_rows.append(item)
            if audit["retain"]:
                retained_rows.append(item)

        strict_pass = (
            len(retained_rows) == N_TOP
            and all(row["audit"]["decision"] == "retain_direct" for row in retained_rows)
        )
        if strict_pass:
            panel_status = "eligible_natural_top10"
        elif retained_rows:
            panel_status = "partial_only_do_not_present_as_naturally_clean"
        else:
            panel_status = "not_eligible_natural_top10"

        entry = {
            "phenotype": label,
            "phecode": phecode,
            "category": categories.get(label, ""),
            "display_title": title_for(label, auc_mean, auc_lo, auc_hi),
            "auroc": {
                "mean": auc_mean,
                "std_across_seeds": auc_std,
                "ci95_mean": [auc_lo, auc_hi],
                "n_seeds": N_SEEDS,
                "ci_method": "two-sided t interval across 20 initialization seeds (df=19)",
                "per_seed": [float(x) for x in aucs],
            },
            "clinical_profile": {
                "source": profile.source,
                "notes": profile.notes,
                "target_block": TARGET_BLOCKS.get(label),
                "ctpa_fov_block": label in CTPA_FOV_BLOCKS,
                "branch_specific_note": BRANCH_SPECIFIC.get(label),
                "imaging_support_note": IMAGING_SUPPORT_ONLY.get(label),
            },
            "audit_summary": {
                "panel_status": panel_status,
                "strict_natural_top10_pass": strict_pass,
                "n_retained": len(retained_rows),
                "n_rejected": N_TOP - len(retained_rows),
                "classifier_tier_counts": dict(sorted(tier_counts.items())),
                "reason_code_counts": dict(sorted(decision_counts.items())),
            },
            "natural_top10": natural_rows,
            "audited_top10": retained_rows,
        }
        entries.append(entry)

    # Put the strongest clinically aligned natural explanations first while
    # retaining all 221 labels, making the selection rule inspectable.
    entries.sort(
        key=lambda e: (
            -int(e["audit_summary"]["strict_natural_top10_pass"]),
            -int(e["audit_summary"]["n_retained"]),
            -float(e["auroc"]["mean"]),
            e["phenotype"],
        )
    )
    strict_passes = [
        {
            "phenotype": e["phenotype"],
            "phecode": e["phecode"],
            "auroc_mean": e["auroc"]["mean"],
            "n_retained": e["audit_summary"]["n_retained"],
        }
        for e in entries
        if e["audit_summary"]["strict_natural_top10_pass"]
    ]
    retained_decisions = Counter(
        concept["audit"]["decision"]
        for entry in entries
        for concept in entry["audited_top10"]
    )

    output = {
        "metadata": {
            "generated_on": date.today().isoformat(),
            "task": "INSPECT-221 phenotype linear probe",
            "method": "f2llm",
            "probe": "Adam linear probe, lr=3e-2, 20 initialization seeds",
            "concept_bank_size": bank_size,
            "concept_embedding_dim": embedding_shape[1],
            "ranking": (
                "Natural positive top-10 recomputed over all 376,194 concepts as "
                "mean_seed(W[label]) dot normalize(raw_embedding - full_bank_mean); "
                "no deduplication and no replacement after rejection."
            ),
            "coefficient": "raw W[label] dot centered and L2-normalized f2llm concept embedding",
            "audit": (
                "Deterministic conservative clinical-radiology screening using affirmative/direct, "
                "associated, exclusion, target-specificity, and CTPA-field-of-view rules."
            ),
            "retained_definition": (
                "Affirmative target-matched CTPA-visible evidence. Retention does not by itself "
                "prove the clinical diagnosis and is not a substitute for radiologist adjudication."
            ),
            "panel_eligibility": (
                "A natural panel passes only when all 10 unmodified natural top concepts receive "
                "retain_direct; imaging-support and composite-branch decisions do not qualify."
            ),
            "phenotype_order": (
                "strict pass first, then retained-count descending, AUROC descending, phenotype ascending; "
                "all 221 phenotypes are included."
            ),
            "title_format": "Phenotype on line 1; AUROC and 95% CI begin on line 2.",
            "source_files": {
                "seed_bundle_pattern": str(SEED_TEMPLATE.relative_to(ROOT)),
                "seed_bundle_count": len(seed_paths),
                "seed_bundle_sha256": {
                    path.name: sha256(path) for path in seed_paths
                },
                "embeddings": str(args.embeddings.relative_to(ROOT)),
                "concept_bank": str(args.concept_bank.relative_to(ROOT)),
                "clinical_rules": "trusted_concept_space.py plus explicit strict policies in this builder",
            },
            "provenance_checks": {
                "all_221_auroc_distributions_recomputed_from_seed_bundles": True,
                "all_221_top10_rankings_recomputed_from_current_seed_weights": True,
                "all_ranked_coefficients_match_per_seed_means": True,
                "embedding_bank_row_count_match": True,
                "bank_mean_l2_norm": float(np.linalg.norm(mean_vector)),
            },
            "clinical_basis": [
                {
                    "principle": "Bronchiectasis requires direct bronchial dilatation/morphologic evidence.",
                    "source": "https://pubmed.ncbi.nlm.nih.gov/34570994/",
                },
                {
                    "principle": "CT pneumonia patterns overlap and do not reliably establish microbial etiology.",
                    "source": "https://pubmed.ncbi.nlm.nih.gov/18835120/",
                },
                {
                    "principle": "Direct pulmonary embolism evidence is an intraluminal pulmonary-arterial filling defect.",
                    "source": "https://pubmed.ncbi.nlm.nih.gov/15371604/",
                },
            ],
        },
        "inspection_summary": {
            "n_phenotypes": len(entries),
            "n_natural_concepts_audited": len(entries) * N_TOP,
            "n_audit_retained_concepts": sum(
                e["audit_summary"]["n_retained"] for e in entries
            ),
            "retained_decision_counts": dict(sorted(retained_decisions.items())),
            "n_strict_natural_top10_passes": len(strict_passes),
            "strict_natural_top10_passes": strict_passes,
            "classifier_tier_counts": dict(sorted(global_tiers.items())),
            "reason_code_counts": dict(sorted(global_decisions.items())),
        },
        "phenotypes": entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(json.dumps(output["inspection_summary"], indent=2))


if __name__ == "__main__":
    main()

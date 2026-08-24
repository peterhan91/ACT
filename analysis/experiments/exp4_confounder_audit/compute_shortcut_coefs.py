#!/usr/bin/env python3
"""Build shortcut-figure data from the f2llm full-concept-bank linear probe.

Probe: INSPECT-221 phenotype linear probe in f2llm LLM-embedding space, trained
with Adam (lr=3e-2) and rerun over 20 seeds:
    outputs/v1/external/phenotype__linear_f2llm__seed{1..20}__lr3e-2__coefs.pt

Concept auditing: for a phenotype row W[label] (a direction in LLM space) and the
full concept bank e (376k concepts, centered over the bank then L2-normalized),
the per-concept *linear-model coefficient* is the raw projection

    coef(concept) = W[label] . e_concept          (W NOT renormalized)

i.e. how strongly the linear probe leans on that concept's direction. Concepts are
ranked by the 20-seed mean coefficient; this script keeps the **raw top-5** with
NO de-duplication, so near-duplicate paraphrases are shown as-is. Each concept
carries its 20-seed mean/std + the 20 per-seed values (scatter). Each phenotype
carries the 20-seed mean +/- 95% CI test AUROC.

Shortcut proxy-theme labels (the right-hand diagram top circle) are derived from
the actually-plotted top-5 concepts (majority vote over a keyword classifier), so
the graph model always describes what the bars show.

Output: results/shortcut_coefs.json (consumed by plot_shortcut.py).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results" / "shortcut_coefs.json"
BANK_NPZ = ROOT / "concept_bank.f2llm_emb.npz"
SEED_GLOB = "outputs/v1/external/phenotype__linear_f2llm__seed{s}__lr3e-2__coefs.pt"
N_SEEDS = 20
N_TOP = 5
TCRIT_19 = 2.093  # t_{0.975, df=19} for the 95% CI half-width

PANELS = [
    # Full-bank examples retained only when the raw top concepts are visible on
    # chest CT and clinically concordant with at least one explicit label branch.
    ("Pneumonia", "aligned"),
    ("Pulmonary collapse; interstitial and compensatory emphysema", "aligned"),
    ("Bronchiectasis", "aligned"),
    ("Shock", "shortcut"),
    ("Anemia in neoplastic disease", "shortcut"),
    ("Secondary malignancy of bone", "shortcut"),
    ("Diaphragmatic hernia", "shortcut"),
    ("Lymphadenitis", "shortcut"),
    ("Other disorders of liver", "shortcut"),
]

AUDIT_NOTES = {
    "Pneumonia": (
        "Imaging-concordant air-space disease; CT supports pneumonia but does not "
        "establish a microbial cause."
    ),
    "Pulmonary collapse; interstitial and compensatory emphysema": (
        "Displayed concepts support the pulmonary-collapse/atelectasis branch of "
        "this composite phenotype label."
    ),
    "Bronchiectasis": "Direct structural airway abnormality visible on chest CT.",
}

# Ordered most-specific first: each plotted concept is matched against these in
# turn, and the panel's proxy-theme label is the majority vote over its top-5
# (ties broken toward the higher-coefficient concept).
THEMES = [
    ("Pericardial effusion / fluid", r"pericard|pneumopericard"),
    ("Aortic / coronary calcification", r"atheroscler|calcif|atheroma|\bcoronary\b|aneurysm"),
    ("Lymph nodes", r"lymph node|lymphaden|lymphangit"),
    ("Consolidation / ground-glass", r"consolidat|ground.?glass|infiltrat|pneumonic|airspace|opacit"),
    ("Pleural effusion / fluid", r"pleural|hydropneumothorax"),
    ("Ascites / free fluid", r"ascites|peritone|abdom|free fluid|intra-?abdominal|\bpelvis\b"),
    ("Atelectasis / collapse", r"atelecta|collapse"),
    ("Effusion / fluid", r"effusion|fluid"),
]


def classify_theme(concept: str) -> str | None:
    c = concept.lower()
    for label, pat in THEMES:
        if re.search(pat, c):
            return label
    return None


def dominant_theme(rows: list[dict]) -> str | None:
    """Majority theme over the plotted (coef-descending) concepts.

    rows are ordered by descending coefficient, so ties resolve toward the
    earliest (highest-coefficient) concept's theme.
    """
    labels = [classify_theme(r["concept"]) for r in rows]
    counts = Counter(l for l in labels if l)
    if not counts:
        return None
    return max(counts, key=lambda l: (counts[l], -labels.index(l)))


def load_seed_W(label_order: list[str]) -> tuple[np.ndarray, dict]:
    """Return Ws (S, n_panels, D) for the panel phenotypes + per-seed test AUCs."""
    Ws, aucs, labels_ref = [], {p: [] for p in label_order}, None
    idx = None
    for s in range(1, N_SEEDS + 1):
        f = ROOT / SEED_GLOB.format(s=s)
        b = torch.load(f, map_location="cpu", weights_only=False)
        if labels_ref is None:
            labels_ref = list(b["labels"])
            idx = [labels_ref.index(p) for p in label_order]
        elif list(b["labels"]) != labels_ref:
            raise ValueError(f"label order mismatch in {f}")
        Ws.append(b["W"].numpy()[idx].astype(np.float64))
        for p in label_order:
            aucs[p].append(float(b["test_per_label_auc"][p]))
    return np.stack(Ws, axis=0), aucs


def bank_mean(emb, chunk: int = 40000) -> np.ndarray:
    M, D = emb.shape
    acc = np.zeros(D, dtype=np.float64)
    for i in range(0, M, chunk):
        acc += emb[i:i + chunk].astype(np.float64).sum(0)
    return (acc / M).astype(np.float32)


def centered_norm(rows: np.ndarray, mean: np.ndarray) -> np.ndarray:
    v = rows.astype(np.float32) - mean
    v /= np.linalg.norm(v, axis=1, keepdims=True).clip(1e-9)
    return v


def main() -> None:
    label_order = [p for p, _ in PANELS]

    Ws, aucs = load_seed_W(label_order)              # (S, P, D)
    Wbar = Ws.mean(axis=0).astype(np.float32)        # (P, D)

    npz = np.load(BANK_NPZ, allow_pickle=True, mmap_mode="r")
    emb = npz["emb"]                                  # (M, D), memory-mapped
    concepts = [str(c) for c in np.load(BANK_NPZ, allow_pickle=True)["concepts"]]
    M = emb.shape[0]
    mean = bank_mean(emb)

    # One pass: 20-seed mean coefficient over the whole bank -> rank -> raw top-5.
    coef_mean = np.zeros((len(label_order), M), dtype=np.float32)
    for i in range(0, M, 40000):
        ec = centered_norm(emb[i:i + 40000], mean)
        coef_mean[:, i:i + 40000] = Wbar @ ec.T
    top_idx = {pi: np.argsort(-coef_mean[pi])[:N_TOP].tolist()
               for pi in range(len(label_order))}

    # Per-seed coefficients for just the selected concepts.
    sel = sorted({c for v in top_idx.values() for c in v})
    ec_sel = centered_norm(emb[sel], mean)           # (n_sel, D)
    sel_pos = {c: r for r, c in enumerate(sel)}
    # per-seed coef[s, pi, concept] = Ws[s, pi] . ec
    perseed = np.einsum("spd,nd->spn", Ws.astype(np.float32), ec_sel)  # (S,P,n_sel)

    out = []
    for pi, (ph, kind) in enumerate(PANELS):
        rows = []
        for ci in top_idx[pi]:
            vals = perseed[:, pi, sel_pos[ci]]
            rows.append({
                "concept_index": int(ci),
                "concept": concepts[ci],
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "per_seed": [float(x) for x in vals],
            })
        au = np.array(aucs[ph], dtype=np.float64)
        auc, auc_std = float(au.mean()), float(au.std(ddof=1))
        half = TCRIT_19 * auc_std / math.sqrt(len(au))
        theme = dominant_theme(rows)
        out.append({
            "phenotype": ph,
            "kind": kind,
            "auc": auc,
            "auc_lo": auc - half,
            "auc_hi": auc + half,
            "auc_std": auc_std,
            "n_seeds": int(len(au)),
            "audit_note": AUDIT_NOTES.get(ph),
            "graph_top": (theme + " (candidate proxy)") if kind == "shortcut" else None,
            "graph_relation": "semantic alignment" if kind == "shortcut" else "direct signal",
            "graph_bottom": ph,
            "source": {
                "probe": "phenotype__linear_f2llm__seed{1..20}__lr3e-2 (Adam, full bank)",
                "coefficient": "raw W[label] . centered+normalized concept embedding",
                "selection": "raw top-5 by 20-seed mean coefficient (NO de-duplication)",
            },
            "concepts": rows,
        })
        print(f"{ph:55s} AUROC {auc:.3f} CI {auc-half:.3f}-{auc+half:.3f} | "
              f"theme={theme or '-':32s} | top={rows[0]['concept'][:42]}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

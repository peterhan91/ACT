#!/usr/bin/env python3
# LEGACY ALIGNMENT CONVENTION. The LLM-cosine branch below (llm_coefs_pt)
# normalizes the probe weight matrix W before scoring, so its output is a
# cosine, NOT the manuscript's Equation 2 alignment score. The authoritative
# raw-unnormalized-weight implementation for Figs. 5-6 lives in
# analysis/experiments/exp4_confounder_audit/ (export_f2llm_adam_top25.py,
# build_f2llm_natural_clinical_audit.py). This script is the CT-RATE-18 CBM
# audit driver used by run_pipeline.sh step 4.
"""
Step 3 — Concept-level audit on the CT-RATE test set.

Two artifacts:
    A) Per-label concept importance (CLEAR's `concept_importance.py` analogue).
       Uses the trained CBM weight matrix W (n_labels × n_concepts). For each
       label we sort by W[label] and write the top-K positive / top-K negative
       concepts. Because our W is *directly* on the concept axis, no extra
       cosine-alignment step is needed.

    B) Per-image top concepts + per-concept top images (image_retriever-style).
       For each test volume, the K most-activated concepts (highest concept_sim).
       For a curated probe-concept set (or the user's own --probe_concepts file),
       the top-K most similar test volumes.

Outputs (in outputs/audit/):
    label_top_concepts.csv         — long table: (label, rank, sign, concept, weight)
    image_top_concepts.parquet     — per (volume, rank) → (concept, score)
    concept_top_images.csv         — per (concept, rank) → (volume, score)
    label_top_concepts.json        — same as csv but nested for the dashboard
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from clear3d.data import load_split
from clear3d.features import encode_images
from clear3d.paths import AUDIT_DIR, CBM_DIR, DATASETS, LABELS_18
from clear3d.projection import load_concept_bank


# A small curated probe set: the 18 label names plus a few canonical free-form
# concepts to demo retrieval. These are stored verbatim and similarity is
# computed against the bank by exact-string match (cheap) + nearest-CLIP-text
# fallback for non-matches.
DEFAULT_PROBE_CONCEPTS = [
    "small pleural effusion",
    "right upper lobe consolidation",
    "ground-glass opacity",
    "centrilobular emphysema",
    "spiculated solid nodule",
    "coronary artery calcifications",
    "mediastinal lymphadenopathy",
    "atelectasis",
    "interlobular septal thickening",
    "mosaic attenuation",
]


def label_top_concepts(coefs_pt: Path, k: int = 100,
                       *, llm_coefs_pt: Path | None = None,
                       llm: str | None = None,
                       concepts: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (long-CSV, nested-JSON) of top-K +/- concepts per label.

    Two modes:
      - CBM-direct (default): concept-axis weights from the CBM's W matrix
        (n_labels × n_concepts). The "weight" is W[label, concept] directly.
      - LLM-cosine (if llm_coefs_pt given): a linear probe trained on the
        LLM-projected representation. We compute cosine alignment between
        W[label] (in LLM space) and each concept's LLM embedding, mirroring
        CLEAR's `concept_importance.py`. Use --llm to pick {sfr, harrier}.
    """
    if llm_coefs_pt is not None:
        from clear3d.projection import load_concept_bank
        bundle = torch.load(llm_coefs_pt, map_location="cpu", weights_only=False)
        W = bundle["W"].numpy()      # (n_labels, llm_dim)
        labels = bundle["labels"]
        bank = load_concept_bank(llm=llm, device="cpu")
        if concepts is None:
            concepts = bank["concepts"]
        llm_emb = bank["llm_emb"].numpy()   # (n_concepts, llm_dim), L2-normalized
        # cosine = (W_norm @ llm_emb.T) per label
        W_norms = np.linalg.norm(W, axis=1, keepdims=True).clip(1e-9)
        W_unit = W / W_norms
        sims = W_unit @ llm_emb.T   # (n_labels, n_concepts)
    else:
        bundle = torch.load(coefs_pt, map_location="cpu", weights_only=False)
        W = bundle["W"].numpy()      # (n_labels, n_concepts)
        if concepts is None:
            concepts = bundle["concepts"]
        labels = bundle["labels"]
        sims = W                      # raw weights treated as alignment

    rows = []
    payload = {}
    for li, lbl in enumerate(labels):
        v = sims[li]
        order_pos = np.argsort(-v)[:k]
        order_neg = np.argsort(v)[:k]
        positive = []
        for r, ci in enumerate(order_pos, start=1):
            rows.append({"label": lbl, "sign": "positive", "rank": r,
                         "concept": concepts[ci], "weight": float(v[ci])})
            positive.append({"rank": r, "concept": concepts[ci], "weight": float(v[ci])})
        negative = []
        for r, ci in enumerate(order_neg, start=1):
            rows.append({"label": lbl, "sign": "negative", "rank": r,
                         "concept": concepts[ci], "weight": float(v[ci])})
            negative.append({"rank": r, "concept": concepts[ci], "weight": float(v[ci])})
        payload[lbl] = {"positive": positive, "negative": negative,
                        "stats": {"max_pos": float(v.max()),
                                  "min_neg": float(v.min())}}
    df = pd.DataFrame(rows)
    return df, payload


@torch.no_grad()
def per_image_top_concepts(img_feats: np.ndarray, clip_emb: torch.Tensor,
                           concepts: list[str], volume_names: list[str],
                           k: int = 20, batch: int = 256) -> pd.DataFrame:
    """For each image, top-k activated concepts."""
    img = torch.from_numpy(img_feats).cuda()
    img = F.normalize(img, dim=-1)
    rows = []
    for s in tqdm(range(0, len(img), batch), desc="per-image topK"):
        e = min(s + batch, len(img))
        sims = (img[s:e] @ clip_emb.T).float()  # (b, M)
        topv, topi = torch.topk(sims, k=k, dim=1)
        topv = topv.cpu().numpy()
        topi = topi.cpu().numpy()
        for bi in range(e - s):
            vol = volume_names[s + bi]
            for r in range(k):
                rows.append({"VolumeName": vol, "rank": r + 1,
                             "concept": concepts[topi[bi, r]],
                             "score": float(topv[bi, r])})
    return pd.DataFrame(rows)


@torch.no_grad()
def per_concept_top_images(probe_concepts: list[str], img_feats: np.ndarray,
                           clip_emb: torch.Tensor, concepts: list[str],
                           volume_names: list[str], k: int = 20) -> pd.DataFrame:
    """For each probe concept, find the top-k most activating test images."""
    concept_to_idx = {c: i for i, c in enumerate(concepts)}
    img = torch.from_numpy(img_feats).cuda()
    img = F.normalize(img, dim=-1)

    rows = []
    for q in probe_concepts:
        if q.lower() in concept_to_idx:
            ci = concept_to_idx[q.lower()]
            ce = clip_emb[ci:ci + 1]                  # (1, 768)
            label_for_query = concepts[ci]
            match_kind = "exact"
        else:
            # Nearest-concept fallback: look up the closest concept by string
            # match (cheap heuristic — substring → first hit).
            cands = [c for c in concepts if q.lower() in c.lower()]
            if not cands:
                rows.append({"query": q, "rank": None, "VolumeName": None,
                             "score": None, "matched_concept": None,
                             "match_kind": "none"})
                continue
            ci = concept_to_idx[cands[0]]
            ce = clip_emb[ci:ci + 1]
            label_for_query = cands[0]
            match_kind = "substring"

        ce = F.normalize(ce, dim=-1)
        sims = (img @ ce.T).squeeze(-1)
        topv, topi = torch.topk(sims, k=min(k, len(sims)))
        topv = topv.cpu().numpy()
        topi = topi.cpu().numpy()
        for r in range(len(topi)):
            rows.append({"query": q, "rank": r + 1,
                         "VolumeName": volume_names[int(topi[r])],
                         "score": float(topv[r]),
                         "matched_concept": label_for_query,
                         "match_kind": match_kind})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coefs", default=str(CBM_DIR / "coefs.pt"))
    ap.add_argument("--llm_coefs", default=None,
                    help="optional linear-probe coefs (e.g. linear_sfr__coefs.pt) for cosine-alignment audit")
    ap.add_argument("--llm", default=None, choices=[None, "sfr", "harrier", "openai"],
                    help="which LLM space the linear-probe coefs are in")
    ap.add_argument("--dataset", default="ctrate_test")
    ap.add_argument("--top_k_per_label", type=int, default=100)
    ap.add_argument("--top_k_per_image", type=int, default=20)
    ap.add_argument("--top_k_per_concept", type=int, default=20)
    ap.add_argument("--probe_concepts_file", default=None,
                    help="optional newline-separated text file of probe concepts")
    ap.add_argument("--out_suffix", default="",
                    help="appended to label_top_concepts.* output filenames")
    args = ap.parse_args()

    # ---------------------------------------------------------------- (A)
    print("=== A. Label-level concept importance ===")
    coef_path = Path(args.coefs)
    llm_coef_path = Path(args.llm_coefs) if args.llm_coefs else None

    if llm_coef_path is not None:
        if not llm_coef_path.exists():
            print(f"  !! linear-probe coefs not found at {llm_coef_path}")
            return
        print(f"  using LLM-space cosine alignment ({args.llm}, {llm_coef_path.name})")
        df_lbl, payload = label_top_concepts(coef_path, k=args.top_k_per_label,
                                             llm_coefs_pt=llm_coef_path, llm=args.llm)
    else:
        if not coef_path.exists():
            print(f"  !! coefs.pt not found at {coef_path}; run exp_cbm_ct.py first")
            return
        print(f"  using CBM-direct weights ({coef_path.name})")
        df_lbl, payload = label_top_concepts(coef_path, k=args.top_k_per_label)

    suf = args.out_suffix
    out_csv = AUDIT_DIR / f"label_top_concepts{suf}.csv"
    out_json = AUDIT_DIR / f"label_top_concepts{suf}.json"
    df_lbl.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved → {out_csv} ({len(df_lbl)} rows, {len(payload)} labels)")

    # ---------------------------------------------------------------- (B)
    print("\n=== B. Per-image top concepts + per-concept retrieval ===")
    df = load_split(args.dataset)
    img_feats = encode_images(args.dataset)
    bank = load_concept_bank(llm="sfr", device="cpu")
    clip_emb = bank["clip_text_emb"].cuda()
    concepts = bank["concepts"]
    del bank

    df_img = per_image_top_concepts(
        img_feats, clip_emb, concepts,
        volume_names=df["VolumeName"].tolist(),
        k=args.top_k_per_image,
    )
    out_img = AUDIT_DIR / "image_top_concepts.parquet"
    df_img.to_parquet(out_img, index=False)
    print(f"  saved → {out_img}  ({len(df_img)} rows)")

    if args.probe_concepts_file:
        probe = [l.strip() for l in open(args.probe_concepts_file) if l.strip()]
    else:
        probe = DEFAULT_PROBE_CONCEPTS
    df_concept = per_concept_top_images(
        probe, img_feats, clip_emb, concepts,
        volume_names=df["VolumeName"].tolist(),
        k=args.top_k_per_concept,
    )
    df_concept.to_csv(AUDIT_DIR / "concept_top_images.csv", index=False)
    print(f"  saved → {AUDIT_DIR/'concept_top_images.csv'}  ({len(df_concept)} rows)")

    print("\n=== Audit done ===")


if __name__ == "__main__":
    main()

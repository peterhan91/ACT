#!/usr/bin/env python3
# =============================================================================
# LEGACY IMPLEMENTATION — do NOT use for manuscript alignment scores.
# This script normalizes the probe weights W into a cosine similarity; it does
# NOT implement manuscript Eq. 2. The authoritative alignment score feeding
# Figs. 5-6 is the raw-unnormalized-weight implementation in
# analysis/experiments/exp4_confounder_audit/ (export_f2llm_adam_top25.py,
# build_f2llm_natural_clinical_audit.py). This file's cosine output is used
# only for top-k subset selection (run_v1_phenotype_lbfgs_f2llm.sh).
# =============================================================================
"""20-seed concept-importance audit (CT-RATE-18 / PE-3 / Phenotype-221).

Loads the 20 per-seed coef bundles for a (task, llm) pair at the task's best
LR, averages the weight matrix W across seeds, and emits the audit JSON/CSV in
the same canonical filenames the dashboards already consume:

    task=ctrate18 → outputs/v1/audit/label_top_concepts.linear_<llm>.{csv,json}
    task=pe       → outputs/v1/audit/pe_pheno__linear_<llm>__concept_importance.{csv,json}
    task=phenotype→ outputs/v1/audit/phenotype__linear_<llm>__concept_importance.{csv,json}

Audit pipeline:
  - linear-probe W is in LLM space (n_labels × llm_dim).
  - Cosine alignment between L2-normalized W[label] and (centered, L2-normalized)
    concept embeddings — matches `cosine_align_audit` in exp_phenotype_ct.py for
    PE-3 / Phenotype-221, and the LLM-cosine branch of concept_audit_ct.py for
    CT-RATE-18.
  - For CT-RATE-18 the historical audit did NOT center concepts; we keep that
    convention (--no_center_concepts implied for task=ctrate18 unless overridden).

Stability sidecar:
  - `<output>__20seeds_meta.json` records, per label, the top-10 positive/negative
    concept lists from each individual seed plus mean ± std of alignment for the
    20-seed top-10. Lets a reviewer judge how stable a top-10 list is across
    seeds without rerunning anything.

Usage:
    python concept_audit_20seeds.py --task ctrate18 --llm openai
    python concept_audit_20seeds.py --task pe       --llm openai
    python concept_audit_20seeds.py --task phenotype --llm openai
    # repeat for --llm harrier / --llm sfr
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clear3d.paths import AUDIT_DIR, CBM_DIR, EXTERNAL_DIR
from clear3d.projection import load_concept_bank


TASKS = {
    "ctrate18": {
        "dir": CBM_DIR,
        "prefix": "linear_{llm}",
        "bestlr_file": "linear_{llm}__bestlr.txt",
        "twenty_csv": "linear_{llm}__20seeds_summary.csv",
        "out_basename": "label_top_concepts.linear_{llm}",
        "weight_field": "weight",
        "default_center": False,
    },
    "pe": {
        "dir": EXTERNAL_DIR,
        "prefix": "pe_pheno__linear_{llm}",
        "bestlr_file": "pe_pheno__linear_{llm}__bestlr.txt",
        "twenty_csv": "pe_pheno__linear_{llm}__20seeds_summary.csv",
        "out_basename": "pe_pheno__linear_{llm}__concept_importance",
        "weight_field": "alignment",
        "default_center": True,
    },
    "phenotype": {
        "dir": EXTERNAL_DIR,
        "prefix": "phenotype__linear_{llm}",
        "bestlr_file": "phenotype__linear_{llm}__bestlr.txt",
        "twenty_csv": "phenotype__linear_{llm}__20seeds_summary.csv",
        "out_basename": "phenotype__linear_{llm}__concept_importance",
        "weight_field": "alignment",
        "default_center": True,
    },
    "rsna2023": {
        "dir": EXTERNAL_DIR,
        "prefix": "rsna2023__linear_{llm}",
        "bestlr_file": "rsna2023__linear_{llm}__bestlr.txt",
        "twenty_csv": "rsna2023__linear_{llm}__20seeds_summary.csv",
        "out_basename": "rsna2023__linear_{llm}__concept_importance",
        "weight_field": "alignment",
        "default_center": True,
    },
}


def _read_bestlr(path: Path) -> str:
    txt = path.read_text().strip()
    if not txt:
        raise SystemExit(f"empty bestlr file: {path}")
    return txt


def _load_seed_W(dir_: Path, prefix: str, lr: str, n_seeds: int = 20) -> tuple[np.ndarray, list[str], dict]:
    Ws, labels_ref, meta = [], None, {}
    missing = []
    for s in range(1, n_seeds + 1):
        f = dir_ / f"{prefix}__seed{s}__lr{lr}__coefs.pt"
        if not f.exists():
            missing.append(f.name)
            continue
        b = torch.load(f, map_location="cpu", weights_only=False)
        W = b["W"].numpy().astype(np.float64)
        if labels_ref is None:
            labels_ref = list(b["labels"])
            meta["llm"] = b.get("llm")
            meta["lr"] = lr
        else:
            assert list(b["labels"]) == labels_ref, f"label mismatch in {f}"
            assert W.shape == Ws[0].shape, f"W shape mismatch in {f}"
        Ws.append(W)
    if missing:
        raise SystemExit(f"missing {len(missing)} seed bundles, e.g. {missing[:3]}")
    arr = np.stack(Ws, axis=0)  # (S, n_labels, llm_dim)
    meta["n_seeds"] = arr.shape[0]
    return arr, labels_ref, meta


def _align(W: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Cosine alignment per (label, concept). W: (...,n_labels,llm_dim), e: (n_concepts, llm_dim)."""
    Wn = W / np.linalg.norm(W, axis=-1, keepdims=True).clip(1e-9)
    return Wn @ e.T


def _topk_payload(sims: np.ndarray, labels: list[str], concepts: list[str],
                  k: int, weight_field: str,
                  per_label_test_auc: dict | None = None,
                  per_seed_topk_vals: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """sims: (n_labels, n_concepts) average alignment.

    per_seed_topk_vals: optional dict {(li, ci): np.ndarray of length S with that
    concept's alignment in each seed} — used to attach per-concept mean/std to the
    payload without materializing the full (S, n_labels, n_concepts) tensor.
    """
    rows, payload = [], {}
    for li, lbl in enumerate(labels):
        v = sims[li]
        pos_idx = np.where(v > 0)[0]
        neg_idx = np.where(v < 0)[0]
        pos_sorted = pos_idx[np.argsort(-v[pos_idx])][:k]
        neg_sorted = neg_idx[np.argsort(v[neg_idx])][:k]

        def _row(rank, ci, sign):
            entry = {"rank": rank, "concept": concepts[ci], weight_field: float(v[ci])}
            if per_seed_topk_vals is not None:
                seed_vals = per_seed_topk_vals.get((li, int(ci)))
                if seed_vals is not None:
                    entry["mean"] = float(seed_vals.mean())
                    entry["std"] = float(seed_vals.std(ddof=1))
            return entry

        positive = [_row(r + 1, ci, "positive") for r, ci in enumerate(pos_sorted)]
        negative = [_row(r + 1, ci, "negative") for r, ci in enumerate(neg_sorted)]
        for r, ci in enumerate(pos_sorted, start=1):
            rows.append({"label": lbl, "sign": "positive", "rank": r,
                         "concept": concepts[ci], weight_field: float(v[ci])})
        for r, ci in enumerate(neg_sorted, start=1):
            rows.append({"label": lbl, "sign": "negative", "rank": r,
                         "concept": concepts[ci], weight_field: float(v[ci])})
        stats = {
            "n_positive": int(len(pos_idx)),
            "n_negative": int(len(neg_idx)),
            "max_alignment": float(v.max()),
            "min_alignment": float(v.min()),
        }
        # CT-RATE legacy stats keys for backwards compat
        if weight_field == "weight":
            stats["max_pos"] = float(v.max())
            stats["min_neg"] = float(v.min())
        if per_label_test_auc is not None and lbl in per_label_test_auc:
            rec = per_label_test_auc[lbl]
            stats["test_auc"] = float(rec["mean"])
            if "std" in rec:
                stats["test_auc_std"] = float(rec["std"])
            if "n_seeds" in rec:
                stats["test_auc_n_seeds"] = int(rec["n_seeds"])
        payload[lbl] = {"positive": positive, "negative": negative, "stats": stats}
    return pd.DataFrame(rows), payload


def _streaming_sims(Ws: np.ndarray, e: np.ndarray, labels: list[str],
                    concepts: list[str], k_payload: int, k_stability: int,
                    weight_field: str):
    """One pass over seeds. Returns:
        sims_mean        (n_labels, n_concepts) float32 — alignment of mean(W)
        per_seed_topk    {(li, ci): np.ndarray[S]} only for top-k_payload pos/neg of sims_mean
        sidecar_perseed  {label -> {per_seed_top_pos: [...], per_seed_top_neg: [...]}}
    Streams per-seed sims; never holds (S, n_labels, n_concepts) at once.
    """
    S, n_labels, _ = Ws.shape
    Wbar = Ws.mean(axis=0)
    sims_mean = _align(Wbar.astype(np.float32), e.astype(np.float32))  # (n_labels, n_concepts)

    # Identify which concept indices we need per-seed values for: top-k_payload pos/neg of sims_mean
    needed = {}  # (li, ci) -> np.zeros(S, float32)
    for li in range(n_labels):
        v = sims_mean[li]
        pos = np.where(v > 0)[0]
        neg = np.where(v < 0)[0]
        ps = pos[np.argsort(-v[pos])][:k_payload]
        ns = neg[np.argsort(v[neg])][:k_payload]
        for ci in ps.tolist() + ns.tolist():
            needed[(li, int(ci))] = np.zeros(S, dtype=np.float32)

    sidecar = {lbl: {"per_seed_top_pos": [], "per_seed_top_neg": []}
               for lbl in labels}
    e32 = e.astype(np.float32)
    for s in range(S):
        sims_s = _align(Ws[s].astype(np.float32), e32)  # (n_labels, n_concepts) float32
        # Pull per-seed values for the cells we'll attach mean/std to
        for (li, ci), buf in needed.items():
            buf[s] = sims_s[li, ci]
        # Per-seed top-k_stability lists for each label
        for li, lbl in enumerate(labels):
            v = sims_s[li]
            pos = np.where(v > 0)[0]
            neg = np.where(v < 0)[0]
            ps = pos[np.argsort(-v[pos])][:k_stability]
            ns = neg[np.argsort(v[neg])][:k_stability]
            sidecar[lbl]["per_seed_top_pos"].append([concepts[c] for c in ps])
            sidecar[lbl]["per_seed_top_neg"].append([concepts[c] for c in ns])
        del sims_s

    # Mean-pooled top for sidecar
    for li, lbl in enumerate(labels):
        v = sims_mean[li]
        pos = np.where(v > 0)[0]
        neg = np.where(v < 0)[0]
        sidecar[lbl]["mean_top_pos"] = [concepts[c] for c in pos[np.argsort(-v[pos])][:k_stability]]
        sidecar[lbl]["mean_top_neg"] = [concepts[c] for c in neg[np.argsort(v[neg])][:k_stability]]

    return sims_mean, needed, {"k": k_stability, "n_seeds": S, "labels": sidecar}


def _per_label_test_auc(twenty_csv: Path) -> dict | None:
    """Return {label: {"mean": float, "std": float, "n_seeds": int}}.

    Reads the 20seeds_summary.csv produced by aggregate_20seeds.py.
    """
    if not twenty_csv.exists():
        return None
    df = pd.read_csv(twenty_csv)
    if "label" not in df.columns or "mean_auc" not in df.columns:
        return None
    has_std = "std_auc" in df.columns
    has_n = "n_seeds" in df.columns
    out = {}
    for _, r in df.iterrows():
        lbl = str(r["label"])
        rec = {"mean": float(r["mean_auc"])}
        if has_std and pd.notna(r["std_auc"]):
            rec["std"] = float(r["std_auc"])
        if has_n and pd.notna(r["n_seeds"]):
            rec["n_seeds"] = int(r["n_seeds"])
        out[lbl] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--llm", required=True,
                    choices=["openai", "harrier", "sfr", "gteqwen2", "f2llm"])
    ap.add_argument("--bestlr", default=None,
                    help="override the bestlr file (string, e.g. 3e-2)")
    ap.add_argument("--top_k_per_label", type=int, default=100)
    ap.add_argument("--top_k_stability", type=int, default=10)
    ap.add_argument("--center_concepts", action="store_true",
                    help="force-center concept embeddings (default depends on task)")
    ap.add_argument("--no_center_concepts", action="store_true",
                    help="force no centering")
    ap.add_argument("--n_seeds", type=int, default=20)
    args = ap.parse_args()

    if args.center_concepts and args.no_center_concepts:
        raise SystemExit("pass at most one of --center_concepts / --no_center_concepts")

    spec = TASKS[args.task]
    dir_ = spec["dir"]
    prefix = spec["prefix"].format(llm=args.llm)
    bestlr = args.bestlr or _read_bestlr(dir_ / spec["bestlr_file"].format(llm=args.llm))
    out_base = AUDIT_DIR / spec["out_basename"].format(llm=args.llm)
    weight_field = spec["weight_field"]
    if args.center_concepts:
        center = True
    elif args.no_center_concepts:
        center = False
    else:
        center = spec["default_center"]

    print(f"=== 20-seed audit | task={args.task} llm={args.llm} bestlr={bestlr} center={center} ===")
    Ws, labels, meta = _load_seed_W(dir_, prefix, bestlr, n_seeds=args.n_seeds)
    print(f"  loaded {meta['n_seeds']} seeds | W = {Ws.shape}")

    bank = load_concept_bank(llm=args.llm, device="cpu")
    concepts = bank["concepts"]
    e = bank["llm_emb"].numpy()  # already L2-normalized
    if center:
        e = e - e.mean(axis=0, keepdims=True)
        e = e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-9)

    sims_mean, per_seed_topk_vals, sidecar_payload = _streaming_sims(
        Ws, e, labels, concepts,
        k_payload=args.top_k_per_label, k_stability=args.top_k_stability,
        weight_field=weight_field,
    )

    test_auc_map = _per_label_test_auc(dir_ / spec["twenty_csv"].format(llm=args.llm))
    df_long, payload = _topk_payload(
        sims_mean, labels, concepts, k=args.top_k_per_label,
        weight_field=weight_field, per_label_test_auc=test_auc_map,
        per_seed_topk_vals=per_seed_topk_vals,
    )

    out_csv = out_base.parent / f"{out_base.name}.csv"
    out_json = out_base.parent / f"{out_base.name}.json"
    df_long.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out_csv} ({len(df_long)} rows, {len(payload)} labels)")
    print(f"  wrote {out_json}")

    sidecar = sidecar_payload
    sidecar.update({"task": args.task, "llm": args.llm, "lr": bestlr,
                    "center_concepts": center, "n_seeds": meta["n_seeds"]})
    side_path = out_base.parent / f"{out_base.name}__20seeds_meta.json"
    with open(side_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"  wrote {side_path}")


if __name__ == "__main__":
    main()

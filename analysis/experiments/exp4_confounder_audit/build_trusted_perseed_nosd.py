#!/usr/bin/env python3
"""Build a trusted-probe importance file using the PLAIN back-projection llm_emb @ w_z
(i.e. WITHOUT the z-score de-standardization 1/sd that the canonical importance uses).

The PLAIN w_z form built here is the convention displayed in the paper's
Figure 6 (plot_grid_2x5.py reads this file's output) and matches the
manuscript's displayed restriction weights:
    importance_k = L2norm(emb[idx])[k] .  w_z            # plain dot with the z-space weight
The w_z/sd form is an internal fitting/summary convention used by
_audit_trusted_bootstrap.py and the superseded plot_top_gains.py default:
    importance_k = L2norm(emb[idx])[k] . (w_z / sd)      # gradient of logit wrt raw feature

Everything is per-phenotype: each phenotype uses its OWN trusted slice of the 376k
bank (its own `indices`, its own probe weights W and the 20 train-bootstrap refits).
Inputs (all already on HF, group `refinement`, pulled to outputs/.../):
    trusted_f2llm_bootstrap_perseed_coefs.npz   <slug>__trusted__{W(21,5120),indices(K),concept(K),tier(K)}
                                                 row0 = full-train point fit, rows1..20 = bootstrap seeds
    trusted_f2llm_bootstrap_perseed_index.json  phenotype-name -> {slug, ...}
    trusted_f2llm_bootstrap_concept_importance.json  (wrapper fields: phecode, auc stats, ...)
    concept_bank.f2llm_emb.npz                   emb (376194, 5120)

Output: trusted_f2llm_bootstrap_concept_importance_perseed_nosd.json — SAME schema as the
canonical perseed json, but positive/negative concepts are re-selected and scored by the
no-/sd importance, with `per_seed` = the 20 bootstrap values. plot_top_gains.py --nosd reads it.

    python experiments/exp4_confounder_audit/build_trusted_perseed_nosd.py [--all]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TDIR = ROOT / "outputs/v1/external/trusted_concept_probe_f2llm_lbfgs"
NPZ = TDIR / "trusted_f2llm_bootstrap_perseed_coefs.npz"
INDEX = TDIR / "trusted_f2llm_bootstrap_perseed_index.json"
SUMMARY = TDIR / "trusted_f2llm_bootstrap_concept_importance.json"
BANK = ROOT / "concept_bank.f2llm_emb.npz"
REFINE = HERE / "results" / "f2llm_refinement_all86_concepts.json"
OUT = TDIR / "trusted_f2llm_bootstrap_concept_importance_perseed_nosd.json"

N_KEEP = 10  # positives/negatives stored per phenotype (plot picks top-5 from positives)


def select_emb(emb_mm, indices: np.ndarray) -> np.ndarray:
    """L2-normalized bank rows for `indices`, read in sorted order for mmap locality."""
    order = np.argsort(indices)
    e_sorted = emb_mm[indices[order]].astype(np.float32)
    e = np.empty_like(e_sorted)
    e[order] = e_sorted
    e /= np.linalg.norm(e, axis=1, keepdims=True).clip(1e-9)
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="all 87 labels (default: 8 figure phenotypes)")
    ap.add_argument("--phecodes", nargs="*", default=None,
                    help="explicit phecodes (resolved to phenotype names via the refinement JSON)")
    args = ap.parse_args()

    index = json.loads(INDEX.read_text())
    summary = json.loads(SUMMARY.read_text())
    z = np.load(NPZ, allow_pickle=True)
    emb_mm = np.load(BANK, mmap_mode="r")["emb"]

    refine = json.loads(REFINE.read_text())
    if args.phecodes:
        by_pc = {str(r["phecode"]): r["phenotype"] for r in refine}
        targets = [by_pc[pc] for pc in args.phecodes if pc in by_pc]
        missing = [pc for pc in args.phecodes if pc not in by_pc]
        if missing:
            print(f"[warn] phecodes not in refinement set (skipped): {missing}")
        # preserve any already-built nosd entries so the file stays a superset
        if OUT.exists():
            try:
                out = json.loads(OUT.read_text())
                print(f"[merge] keeping {len(out)} existing nosd entries")
            except Exception:
                out = {}
        else:
            out = {}
    elif args.all:
        targets = [k for k in summary if "trusted" in summary[k]]
        out = {}
    else:
        targets = [r["phenotype"] for r in sorted(refine, key=lambda z_: -z_["auc_gain"])[:8]]
        out = {}

    gmax = 0.0
    for ph in targets:
        meta = index.get(ph)
        if not meta:
            print(f"[skip] {ph}: not in index")
            continue
        slug = meta["slug"]
        W = z[f"{slug}__trusted__W"].astype(np.float32)          # (21,5120) row0=point, 1..20=seeds
        indices = z[f"{slug}__trusted__indices"].astype(np.int64)
        concepts = [str(c) for c in z[f"{slug}__trusted__concept"]]
        tiers = [str(t) for t in z[f"{slug}__trusted__tier"]]
        e = select_emb(emb_mm, indices)                          # (K,5120)
        IMP = e @ W.T                                            # (K,21)  PLAIN llm_emb @ w_z
        point = IMP[:, 0]
        boot = IMP[:, 1:]                                        # (K,20)
        bmean = boot.mean(1)
        bstd = boot.std(1, ddof=1)
        sign_pt = np.sign(point)
        sign_cons = (np.sign(boot) == sign_pt[:, None]).mean(1)
        gmax = max(gmax, float(np.abs(point).max()))

        def pack(ks):
            rows = []
            for rank, k in enumerate(ks, 1):
                rows.append({
                    "concept": concepts[k], "concept_index": int(indices[k]),
                    "tier": tiers[k], "rank": rank,
                    "importance_point": float(point[k]),
                    "importance_boot_mean": float(bmean[k]),
                    "importance_boot_std": float(bstd[k]),
                    "sign_consistency": float(sign_cons[k]),
                    "per_seed": [float(v) for v in boot[k]],
                })
            return rows

        pos = list(np.argsort(-point)[:N_KEEP])
        neg = list(np.argsort(point)[:N_KEEP])
        node = json.loads(json.dumps(summary[ph]))   # keep wrapper fields (phecode, auc, ...)
        node["trusted"]["positive"] = pack(pos)
        node["trusted"]["negative"] = pack(neg)
        node["trusted"]["importance_formula"] = "llm_emb @ w_z (no 1/sd)"
        out[ph] = node
        print(f"{ph:42s} K={len(indices):6d}  top1={concepts[pos[0]][:38]!r}  "
              f"pt={point[pos[0]]:.3f}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}  ({len(out)} phenotypes)")
    print(f"max |point importance| across phenotypes = {gmax:.3f}   (use to set axis unit)")


if __name__ == "__main__":
    main()

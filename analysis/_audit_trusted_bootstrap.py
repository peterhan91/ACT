#!/usr/bin/env python3
"""C=1e6 LBFGS trusted-concept probe + audit, WITH train bootstrap.

Faithful re-run of the OLD unregularized setting
(`_audit_trusted_openai_probes.py` / `trusted_concept_probe.py --solver lbfgs
--lbfgs_C 1e6`) — same data, same z-scored concept-restricted LLM projection,
same `LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)` — with ONE
addition: each per-label probe is refit on a TRAIN bootstrap resample
(`np.random.default_rng(seed).integers(0, N, N)`), seeds 1..n_seeds, plus the
full-train POINT fit. LBFGS is deterministic, so the bootstrap is the only
source of variance (matches the convention in exp_phenotype_ct.py:337-341 and
refine_probe_llmspace.py).

For every (label, method) it reports
  * AUROC      : point + bootstrap mean / std / 2.5-97.5 percentile CI / min
  * importance : per-concept signed (w_z/sd).llm_emb, point + bootstrap
                 mean / std + sign-stability (fraction of seeds matching the
                 point sign) — the bootstrap directly quantifies whether the
                 C=1e6 weights give a *stable* concept attribution on the
                 curated subspace, or the known weight-explosion garbage.

methods = original_topk (no purification, RANK top-100), trusted (purified),
direct (direct-tier only). Process-parallel across labels (joblib loky,
single-thread BLAS) to saturate the 72 Grace cores.

Outputs (under outputs/v1/external/trusted_concept_probe_<llm>_lbfgs/):
  trusted_<llm>_bootstrap_auroc.csv
  trusted_<llm>_bootstrap_concept_importance.csv
  trusted_<llm>_bootstrap_concept_importance.json
  trusted_<llm>_bootstrap_meta.json
"""
from __future__ import annotations
import argparse, json, os, time, warnings
os.environ.setdefault("CLIP3D_RUN_TAG", "v1")
# Single-thread BLAS so process-level parallelism scales linearly.
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from clear3d.paths import (CACHE, PHENOTYPE_LABELS, PHENOTYPE_MANIFEST,
                           PHENOTYPE_NAMES_CSV, ROOT)
from trusted_concept_space import slug

TOP_K_POS = 10
TOP_K_NEG = 5


def norm_rows(x):
    x = x.astype(np.float32, copy=False)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def load_splits():
    splits = ["inspect_train", "inspect_valid", "inspect_test"]
    img, vols = [], []
    for s in splits:
        idx = pd.read_csv(CACHE / f"volume_index.{s}.csv")
        img.append(np.load(CACHE / f"img_feats.{s}.npy"))
        vols.append(idx["VolumeName"].tolist())
    return {
        "img": np.concatenate(img, 0),
        "vol_to_row": {v: i for i, v in enumerate(sum(vols, []))},
    }


def gather(manifest_split, labels_df, label_cols, lookup):
    manifest_split = manifest_split[manifest_split["visit_occurrence_id"].notna()]
    rows = [(v, lookup["vol_to_row"][v]) for v in manifest_split["VolumeName"]
            if v in lookup["vol_to_row"] and v in labels_df.index]
    vols, src = zip(*rows)
    src = np.asarray(src, dtype=np.int64)
    return {
        "vols": list(vols),
        "img": lookup["img"][src],
        "y": labels_df.loc[list(vols)][label_cols].values.astype(np.float32),
    }


def llm_projection(img_n, clip_emb_sel, llm_emb_sel):
    sims = img_n @ clip_emb_sel.T
    proj = (sims @ llm_emb_sel).astype(np.float32, copy=False)
    return norm_rows(proj)


def fit_one(xtr_z, ytr, xte_z, yte, C, max_iter):
    """One C=1e6 LBFGS fit -> (test_auc, w_z). None if train is single-class."""
    if len(np.unique(ytr)) < 2:
        return None
    m = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter)
    m.fit(xtr_z, ytr.astype(int))
    auc = float(roc_auc_score(yte, m.predict_proba(xte_z)[:, 1]))
    return auc, m.coef_.reshape(-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="f2llm", choices=["openai", "f2llm"])
    ap.add_argument("--methods", nargs="+",
                    default=["original_topk", "trusted", "direct"],
                    choices=["original_topk", "trusted", "direct"])
    ap.add_argument("--C", type=float, default=1e6, help="LBFGS inverse-L2 (old setting = 1e6).")
    ap.add_argument("--max_iter", type=int, default=5000)
    ap.add_argument("--n_seeds", type=int, default=20, help="train-bootstrap resamples (+ 1 point fit).")
    ap.add_argument("--original_top_k", type=int, default=100)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--max_labels", type=int, default=0, help="debug: cap #labels (0=all).")
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()
    llm = args.llm
    C, MAXIT, NSEED, TOPK = args.C, args.max_iter, args.n_seeds, args.original_top_k

    trusted_dir = ROOT / "outputs/v1/audit/trusted_concept_spaces"
    summary = pd.read_csv(trusted_dir / "summary.csv")
    summary_by_label = {r["label"]: dict(r) for _, r in summary.iterrows()}

    rank_path = ROOT / f"outputs/v1/audit/phenotype__linear_{llm}__concept_importance.json"
    rank = json.load(open(rank_path)) if "original_topk" in args.methods else {}

    labels_df = pd.read_parquet(PHENOTYPE_LABELS)
    manifest = pd.read_parquet(PHENOTYPE_MANIFEST)
    phen = pd.read_csv(PHENOTYPE_NAMES_CSV, dtype=str).assign(
        phecode=lambda d: d["phecode"].str.strip(),
        phecode_str=lambda d: d["phecode_str"].str.strip())
    name_to_phecode = dict(zip(phen["phecode_str"], phen["phecode"]))

    lookup = load_splits()
    g = {}
    for split, lbl in [("train", "train"), ("test", "test")]:
        g[split] = gather(manifest[manifest["split"] == lbl], labels_df,
                          sorted(name_to_phecode.values()), lookup)
        g[split]["img"] = norm_rows(g[split]["img"])

    bank_llm = np.load(ROOT / f"concept_bank.{llm}_emb.npz", allow_pickle=True)
    bank_clip = np.load(ROOT / "concept_bank.clip_text_emb.v1.npz", allow_pickle=True)
    concepts_all = list(map(str, bank_llm["concepts"]))
    assert concepts_all == list(map(str, bank_clip["concepts"]))
    clip_emb_full = norm_rows(bank_clip["emb"].astype(np.float32))
    llm_emb_full = norm_rows(bank_llm["emb"].astype(np.float32))
    concept_to_idx = {c: i for i, c in enumerate(concepts_all)}

    phecodes = sorted(name_to_phecode.values())
    label_pos = {p: i for i, p in enumerate(phecodes)}

    out_dir = ROOT / f"outputs/v1/external/trusted_concept_probe_{llm}_lbfgs"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_summ = pd.read_csv(out_dir / "trusted_concept_probe_summary.csv")
    ok_labels = sorted(set(run_summ[run_summ["status"] == "ok"]["label"].tolist()))
    if args.max_labels:
        ok_labels = ok_labels[:args.max_labels]
    print(f"[{llm}] C={C:g} bootstrap n_seeds={NSEED} | {len(ok_labels)} labels "
          f"| methods={args.methods} | cores={os.cpu_count()}", flush=True)

    def concept_indices(label, method):
        """Replicate trusted_concept_probe.py's concept-set construction."""
        if method == "original_topk":
            out = []
            for row in rank.get(label, {}).get("positive", [])[:TOPK]:
                j = concept_to_idx.get(str(row["concept"]))
                if j is not None:
                    out.append(j)
            # dedup, keep order
            seen, ded = set(), []
            for j in out:
                if j not in seen:
                    seen.add(j); ded.append(j)
            return ded, ["original"] * len(ded)
        # trusted / direct: read the curated csv
        raw = summary_by_label.get(label, {}).get("trusted_csv")
        csv_path = Path(raw) if raw and Path(raw).exists() else \
            trusted_dir / f"{slug(label)}.trusted_concepts.csv"
        if not csv_path.exists():
            return [], []
        df = pd.read_csv(csv_path)
        if method == "direct":
            df = df[df["tier"].astype(str) == "direct"]
        df = df.sort_values(by=["score", "tier", "concept"],
                            ascending=[False, True, True]).drop_duplicates(subset=["concept_index"])
        return df["concept_index"].astype(int).tolist(), df["tier"].astype(str).tolist()

    def process_label(label_i, label):
        phc = name_to_phecode[label]
        col_i = label_pos[phc]
        ytr = g["train"]["y"][:, col_i]
        yte = g["test"]["y"][:, col_i]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return None
        N = len(ytr)
        boot_idx = [np.random.default_rng(s).integers(0, N, N) for s in range(1, NSEED + 1)]

        out = {}
        for method in args.methods:
            idxs, tiers = concept_indices(label, method)
            if not idxs:
                continue
            idx_arr = np.asarray(idxs)
            clip_sel = clip_emb_full[idx_arr]
            llm_sel = llm_emb_full[idx_arr]
            concept_labels = [concepts_all[i] for i in idxs]
            xtr = llm_projection(g["train"]["img"], clip_sel, llm_sel)
            xte = llm_projection(g["test"]["img"], clip_sel, llm_sel)

            def auc_and_importance(tr_idx):
                xb = xtr if tr_idx is None else xtr[tr_idx]
                yb = ytr if tr_idx is None else ytr[tr_idx]
                mu = xb.mean(0, keepdims=True); sd = xb.std(0, keepdims=True) + 1e-8
                r = fit_one((xb - mu) / sd, yb, (xte - mu) / sd, yte, C, MAXIT)
                if r is None:
                    return None
                auc, w_z = r
                importance = llm_sel @ (w_z / sd.reshape(-1))   # (K,) signed per-concept
                return auc, importance.astype(np.float32)

            point = auc_and_importance(None)
            if point is None:
                continue
            auc_pt, imp_pt = point
            boots = [auc_and_importance(bi) for bi in boot_idx]
            boots = [b for b in boots if b is not None]
            aucs = np.array([b[0] for b in boots], dtype=np.float64)
            imps = np.stack([b[1] for b in boots]) if boots else np.zeros((0, len(idxs)), np.float32)

            lo, hi = (np.percentile(aucs, [2.5, 97.5]) if len(aucs) else (np.nan, np.nan))
            imp_mean = imps.mean(0) if len(imps) else np.full(len(idxs), np.nan, np.float32)
            imp_std = imps.std(0) if len(imps) else np.full(len(idxs), np.nan, np.float32)
            sign_pt = np.sign(imp_pt)
            sign_cons = ((np.sign(imps) == sign_pt).mean(0) if len(imps)
                         else np.full(len(idxs), np.nan, np.float32))

            order_pos = np.argsort(-imp_pt)[:TOP_K_POS]
            order_neg = np.argsort(imp_pt)[:TOP_K_NEG]

            def entry(rank_i, j):
                return {"rank": rank_i + 1, "concept_index": int(idxs[j]),
                        "concept": concept_labels[j], "tier": tiers[j],
                        "importance_point": float(imp_pt[j]),
                        "importance_boot_mean": float(imp_mean[j]),
                        "importance_boot_std": float(imp_std[j]),
                        "sign_consistency": float(sign_cons[j])}

            out[method] = {
                "phecode": phc, "n_concepts": len(idxs), "n_seeds_used": int(len(aucs)),
                "test_auc_point": float(auc_pt),
                "auc_boot_mean": float(aucs.mean()) if len(aucs) else float("nan"),
                "auc_boot_std": float(aucs.std()) if len(aucs) else float("nan"),
                "auc_ci_lo": float(lo), "auc_ci_hi": float(hi),
                "auc_boot_min": float(aucs.min()) if len(aucs) else float("nan"),
                "positive": [entry(r, j) for r, j in enumerate(order_pos)],
                "negative": [entry(r, j) for r, j in enumerate(order_neg)],
            }
        if not out:
            return None
        print(f"[{label_i+1:03d}/{len(ok_labels):03d}] {label[:48]:<48s} "
              + "  ".join(f"{m}:auc={out[m]['test_auc_point']:.3f}"
                         f"(boot {out[m]['auc_boot_mean']:.3f}±{out[m]['auc_boot_std']:.3f})"
                         for m in out), flush=True)
        return {"label": label, "result": out}

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(process_label)(i, lbl) for i, lbl in enumerate(ok_labels))
    wall = time.time() - t0
    print(f"\nTotal wall time: {wall:.1f}s over {len(ok_labels)} labels", flush=True)

    auc_rows, imp_rows, imp_json = [], [], {}
    for r in results:
        if r is None:
            continue
        label = r["label"]; imp_json[label] = {}
        for method, d in r["result"].items():
            auc_rows.append({"label": label, "phecode": d["phecode"], "method": method,
                             "n_concepts": d["n_concepts"], "n_seeds_used": d["n_seeds_used"],
                             "test_auc_point": d["test_auc_point"],
                             "auc_boot_mean": d["auc_boot_mean"], "auc_boot_std": d["auc_boot_std"],
                             "auc_ci_lo": d["auc_ci_lo"], "auc_ci_hi": d["auc_ci_hi"],
                             "auc_boot_min": d["auc_boot_min"]})
            imp_json[label][method] = {k: d[k] for k in
                ("phecode", "n_concepts", "n_seeds_used", "test_auc_point",
                 "auc_boot_mean", "auc_boot_std", "auc_ci_lo", "auc_ci_hi",
                 "positive", "negative")}
            for sign in ("positive", "negative"):
                for e in d[sign]:
                    imp_rows.append({"label": label, "phecode": d["phecode"], "method": method,
                                     "sign": sign, **e})

    sfx = args.out_suffix
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(out_dir / f"trusted_{llm}_bootstrap_auroc{sfx}.csv", index=False)
    pd.DataFrame(imp_rows).to_csv(out_dir / f"trusted_{llm}_bootstrap_concept_importance{sfx}.csv", index=False)
    with open(out_dir / f"trusted_{llm}_bootstrap_concept_importance{sfx}.json", "w") as f:
        json.dump(imp_json, f, indent=2)
    meta = {"llm": llm, "C": C, "solver": "lbfgs", "max_iter": MAXIT, "n_seeds": NSEED,
            "bootstrap": "train resample default_rng(seed).integers(0,N,N), seeds 1..n_seeds",
            "methods": args.methods, "original_top_k": TOPK, "zscore": "per-resample train stats",
            "n_labels": len(auc_df["label"].unique()) if len(auc_df) else 0,
            "wall_seconds": round(wall, 1), "n_jobs": args.n_jobs}
    with open(out_dir / f"trusted_{llm}_bootstrap_meta{sfx}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== overall mean test AUROC by method (point | bootstrap mean) ===")
    if len(auc_df):
        for m, dm in auc_df.groupby("method"):
            print(f"  {m:<16s} n={len(dm):3d}  point={dm['test_auc_point'].mean():.4f}  "
                  f"boot_mean={dm['auc_boot_mean'].mean():.4f}  "
                  f"mean_boot_std={dm['auc_boot_std'].mean():.4f}")
    print(f"\nWrote trusted_{llm}_bootstrap_auroc{sfx}.csv "
          f"+ concept_importance{sfx}.(csv|json) + meta to {out_dir}")


if __name__ == "__main__":
    main()

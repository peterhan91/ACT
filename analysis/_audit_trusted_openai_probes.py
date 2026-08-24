#!/usr/bin/env python3
"""Concept audit for the trusted-<llm>-LBFGS probes (default llm=openai).

For each label with a non-empty trusted concept space, refit the per-label
binary LBFGS probe (deterministic; reproduces the main run), then rank the
trusted concepts by importance:

    importance(concept j) = (w_z / sd) . llm_emb_j

where w_z is the probe weight in z-scored LLM space and sd is the
per-dim train standard deviation. Positive values = the probe assigns
positive logit when the image expresses that concept.

Output (under outputs/v1/external/trusted_concept_probe_<llm>_lbfgs/):
  - trusted_<llm>_concept_importance.json   per-label top-K signed lists
  - trusted_<llm>_concept_importance.csv    long-form for downstream
"""
from __future__ import annotations
import argparse, json, os, sys, time, warnings
os.environ.setdefault("CLIP3D_RUN_TAG", "v1")
# Force single-thread BLAS so process-level parallelism scales linearly.
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

from clear3d.paths import CACHE, PHENOTYPE_LABELS, PHENOTYPE_MANIFEST, PHENOTYPE_NAMES_CSV, ROOT
from trusted_concept_space import slug


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="openai", choices=["openai", "f2llm"],
                    help="LLM concept-embedding space whose trusted probes to audit.")
    args = ap.parse_args()
    llm = args.llm

    trusted_dir = ROOT / "outputs/v1/audit/trusted_concept_spaces"
    summary = pd.read_csv(trusted_dir / "summary.csv")
    summary_by_label = {r["label"]: dict(r) for _, r in summary.iterrows()}

    labels_df = pd.read_parquet(PHENOTYPE_LABELS)
    manifest = pd.read_parquet(PHENOTYPE_MANIFEST)
    phen = pd.read_csv(PHENOTYPE_NAMES_CSV, dtype=str).assign(
        phecode=lambda d: d["phecode"].str.strip(),
        phecode_str=lambda d: d["phecode_str"].str.strip())
    name_to_phecode = dict(zip(phen["phecode_str"], phen["phecode"]))

    lookup = load_splits()
    g = {}
    for split, lbl in [("train", "train"), ("valid", "validation"), ("test", "test")]:
        g[split] = gather(manifest[manifest["split"] == lbl], labels_df,
                           sorted(name_to_phecode.values()), lookup)
        g[split]["img"] = norm_rows(g[split]["img"])

    bank_llm = np.load(ROOT / f"concept_bank.{llm}_emb.npz", allow_pickle=True)
    bank_clip = np.load(ROOT / "concept_bank.clip_text_emb.v1.npz", allow_pickle=True)
    concepts_all = list(map(str, bank_llm["concepts"]))
    assert concepts_all == list(map(str, bank_clip["concepts"]))
    clip_emb_full = norm_rows(bank_clip["emb"].astype(np.float32))
    llm_emb_full = norm_rows(bank_llm["emb"].astype(np.float32))

    phecodes = sorted(name_to_phecode.values())
    label_pos = {p: i for i, p in enumerate(phecodes)}

    out_dir = ROOT / f"outputs/v1/external/trusted_concept_probe_{llm}_lbfgs"
    # Load run summary to know which labels actually have probes
    run_summ = pd.read_csv(out_dir / "trusted_concept_probe_summary.csv")
    ok_labels = run_summ[run_summ["status"] == "ok"]["label"].tolist()
    print(f"[{llm}] refitting + auditing {len(ok_labels)} labels", flush=True)
    TOP_K_POS = 10
    TOP_K_NEG = 5

    def process_label(label_i: int, label: str):
        phc = name_to_phecode[label]
        col_i = label_pos[phc]
        ytr = g["train"]["y"][:, col_i]
        yte = g["test"]["y"][:, col_i]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return None

        csv_path = trusted_dir / Path(summary_by_label[label]["trusted_csv"]).name
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        df = df.sort_values(by=["score", "tier", "concept"], ascending=[False, True, True])
        df = df.drop_duplicates(subset=["concept_index"])
        idxs = df["concept_index"].astype(int).tolist()
        if not idxs:
            return None
        K = len(idxs)

        idx_arr = np.asarray(idxs)
        clip_sel = clip_emb_full[idx_arr]
        llm_sel = llm_emb_full[idx_arr]

        t0 = time.time()
        xtr = llm_projection(g["train"]["img"], clip_sel, llm_sel)
        xte = llm_projection(g["test"]["img"], clip_sel, llm_sel)

        mu = xtr.mean(0, keepdims=True)
        sd = xtr.std(0, keepdims=True) + 1e-8
        xtr_z = (xtr - mu) / sd
        xte_z = (xte - mu) / sd

        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
        m.fit(xtr_z, ytr.astype(int))
        test_auc = roc_auc_score(yte, m.predict_proba(xte_z)[:, 1])

        w_z = m.coef_.reshape(-1).astype(np.float32)
        w_orig = w_z / sd.reshape(-1)
        concept_scores = llm_sel @ w_orig
        concept_labels = [concepts_all[i] for i in idxs]
        tiers = df["tier"].astype(str).tolist()

        order_pos = np.argsort(-concept_scores)[:TOP_K_POS]
        order_neg = np.argsort(concept_scores)[:TOP_K_NEG]
        pos_list = [{"rank": r+1, "concept_index": int(idxs[j]), "concept": concept_labels[j],
                     "tier": tiers[j], "score": float(concept_scores[j])}
                    for r, j in enumerate(order_pos)]
        neg_list = [{"rank": r+1, "concept_index": int(idxs[j]), "concept": concept_labels[j],
                     "tier": tiers[j], "score": float(concept_scores[j])}
                    for r, j in enumerate(order_neg)]
        elapsed = time.time() - t0
        print(f"[{label_i+1:03d}/{len(ok_labels):03d}] {label[:55]:<55s}  K={K:>5d}  test_auc={test_auc:.4f}  ({elapsed:.1f}s)", flush=True)
        return {
            "label": label, "phecode": phc, "n_concepts": K, "test_auc": float(test_auc),
            "positive": pos_list, "negative": neg_list,
            "w_z": w_z, "b_z": float(m.intercept_[0]),
            "mu": mu.reshape(-1).astype(np.float32),
            "sd": sd.reshape(-1).astype(np.float32),
            "indices": np.asarray(idxs, dtype=np.int64),
        }

    t_start = time.time()
    results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
        delayed(process_label)(i, lbl) for i, lbl in enumerate(ok_labels)
    )
    print(f"Total wall time: {time.time()-t_start:.1f}s", flush=True)

    importance_json = {}
    long_rows = []
    npz_payload = {}
    slug_map = {}
    for r in results:
        if r is None: continue
        label = r["label"]
        importance_json[label] = {
            k: v for k, v in r.items()
            if k in ("phecode", "n_concepts", "test_auc", "positive", "negative")
        }
        for sign in ("positive", "negative"):
            for entry in r[sign]:
                long_rows.append({
                    "label": label, "phecode": r["phecode"], "sign": sign,
                    "rank": entry["rank"], "concept_index": entry["concept_index"],
                    "concept": entry["concept"], "tier": entry["tier"],
                    "importance": entry["score"],
                })
        # Persist probe state per label (slug-keyed in a single NPZ).
        s = slug(label)
        slug_map[label] = s
        npz_payload[f"{s}__w_z"] = r["w_z"]
        npz_payload[f"{s}__b_z"] = np.asarray(r["b_z"], dtype=np.float32)
        npz_payload[f"{s}__mu"] = r["mu"]
        npz_payload[f"{s}__sd"] = r["sd"]
        npz_payload[f"{s}__indices"] = r["indices"]
    np.savez_compressed(out_dir / f"trusted_{llm}_coefs.npz", **npz_payload)
    with (out_dir / f"trusted_{llm}_coefs_label_to_slug.json").open("w") as f:
        json.dump(slug_map, f, indent=2)

    with open(out_dir / f"trusted_{llm}_concept_importance.json", "w") as f:
        json.dump(importance_json, f, indent=2)
    pd.DataFrame(long_rows).to_csv(out_dir / f"trusted_{llm}_concept_importance.csv", index=False)
    print(f"\nWrote {out_dir / f'trusted_{llm}_concept_importance.json'}")
    print(f"Wrote {out_dir / f'trusted_{llm}_concept_importance.csv'}")


if __name__ == "__main__":
    main()

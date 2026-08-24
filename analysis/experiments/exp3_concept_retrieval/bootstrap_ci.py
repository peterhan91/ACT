#!/usr/bin/env python3
"""
exp3 — PATIENT-clustered bootstrap 95% CIs for ALL concept-retrieval metrics, both
directions, for CT-RATE / RSNA / PMBB (radchest dropped per scope).

WHY PATIENT-LEVEL. CT-RATE has up to 16 reconstructions/patient (avg 2.56) and RSNA up
to 2 (avg 1.5); volumes within a patient are correlated, so a volume-level bootstrap
would treat them as independent and report falsely narrow CIs. We resample PATIENTS
with replacement (include all their volumes) — a cluster bootstrap. PMBB is 1
scan/patient, so there it reduces to the ordinary bootstrap.

PROTOCOL. B=1000 patient resamples, SHARED across all 7 models per dataset (so model
differences are PAIRED). Per (model, dataset, metric): point estimate (full sample) +
2.5/97.5 percentile CI, and the paired Δ vs ours (Δ CI excluding 0 => significant).
Concepts are a FIXED vocabulary -> not resampled.

ALL metrics, matching the summary tables:
  concept->image (macro over concepts): pooled R@{1,5,10}, pooled MRR, full-pool
    Precision@{1,5,10}, AP, AUROC.
  image->concept (macro over images, z-scored): Precision@{1,3,5}, Recall@{1,3,5},
    per-image AP (mAP), per-image AUROC.

EFFICIENCY. Every concept->image metric for a replicate is derived from ONE descending
sort per concept: P@k and AP from cumulative positives; AUROC from #negatives-ranked-
below; and the analytic pooled R@k / MRR from q_p = (#negatives above positive p)/nneg
via a precomputed q->metric lookup grid (exact binomial tail, no Monte-Carlo pools).
image->concept metrics are per-image vectors computed once, then averaged over the
resampled images each replicate.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.environ.setdefault("CLEAR3D_REPO", "/path/to/ACT/analysis")
EXP1 = str(Path(REPO) / "experiments" / "exp1_zeroshot")
for p in (REPO, EXP1):
    if p not in sys.path:
        sys.path.insert(0, p)
import common  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "retrieval_results"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {  # f-VLM excluded (not row/col alignable — see concept_retrieval_recall.py)
    "ours": ("ours/results", "plain"), "merlin": ("merlin/results", "native"),
    "ctclip": ("ctclip/results", "native"), "m3dclip": ("m3dclip/results", "native"),
    "biomedclip": ("biomedclip/results", "native"), "medsiglip": ("medsiglip/results", "native"),
    "openai_clip": ("openai_clip/results", "native"),
}
DATASETS = ["ctrate_test", "rsna2023_test", "pmbb_chest_nc", "pmbb_abd_ce"]
P = 64
KS_C2I = (1, 5, 10)          # pooled R@k and full-pool P@k
KS_I2C = (1, 3, 5)
C2I_METRICS = ([f"pooled_R@{k}" for k in KS_C2I] + ["pooled_mrr"]
               + [f"P@{k}" for k in KS_C2I] + ["AP", "AUROC"])
I2C_METRICS = ([f"P@{k}" for k in KS_I2C] + [f"R@{k}" for k in KS_I2C] + ["mAP", "AUROC_img"])

# ---- q -> {pooled R@k, MRR} lookup grid (exact binomial tail; q=frac neg above pos) ----
_QG = np.linspace(0.0, 1.0, 4001)
_GRID = {}
for _k in KS_C2I:
    _b = np.array([comb(P - 1, j) for j in range(_k)], float)
    _GRID[f"pooled_R@{_k}"] = (_b * np.power.outer(_QG, np.arange(_k))
                               * np.power.outer(1 - _QG, (P - 1) - np.arange(_k))).sum(1)
_bm = np.array([comb(P - 1, j) for j in range(P)], float) / (np.arange(P) + 1.0)
_GRID["pooled_mrr"] = (_bm * np.power.outer(_QG, np.arange(P))
                       * np.power.outer(1 - _QG, (P - 1) - np.arange(P))).sum(1)


def patient_ids(dataset, df):
    if dataset.startswith("ctrate"):
        return df["VolumeName"].astype(str).str.extract(r"^(valid_\d+|train_\d+)")[0].values
    if dataset.startswith("rsna"):
        return df["patient_id"].astype(str).values
    if dataset.startswith("pmbb"):
        return df["patient"].astype(str).values
    return df.iloc[:, 0].astype(str).values


def boot_index_sets(pid, B, seed):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(pid, return_inverse=True)
    vols_by_pat = [np.flatnonzero(inv == u) for u in range(len(uniq))]
    return [np.concatenate([vols_by_pat[p] for p in rng.integers(len(uniq), size=len(uniq))])
            for _ in range(B)]


def c2i_concept(s, yj):
    """All concept->image metrics for ONE concept from a single descending sort."""
    npos = int(yj.sum()); M = len(yj); nneg = M - npos
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(-s, kind="stable")
    yo = yj[order].astype(np.float64)
    cum_pos = np.cumsum(yo)
    cum_neg = np.cumsum(1.0 - yo)
    ranks = np.arange(1, M + 1)
    out = {}
    for k in KS_C2I:
        out[f"P@{k}"] = float(cum_pos[k - 1] / k)            # full-pool precision@k
    out["AP"] = float((cum_pos / ranks * yo).sum() / npos)   # average precision
    neg_below = nneg - cum_neg                               # negatives ranked below each pos
    out["AUROC"] = float(neg_below[yo == 1].sum() / (npos * nneg))
    q = cum_neg[yo == 1] / nneg                              # frac negatives ABOVE each positive
    for name in (f"pooled_R@{k}" for k in KS_C2I):
        out[name] = float(np.interp(q, _QG, _GRID[name]).mean())
    out["pooled_mrr"] = float(np.interp(q, _QG, _GRID["pooled_mrr"]).mean())
    return out


def c2i_macro(probs, y, idx):
    s = probs[idx]; yy = y[idx]
    acc = {m: [] for m in C2I_METRICS}
    for j in range(yy.shape[1]):
        r = c2i_concept(s[:, j], yy[:, j])
        if r is None:
            continue
        for m in C2I_METRICS:
            acc[m].append(r[m])
    return {m: float(np.mean(acc[m])) for m in C2I_METRICS}


def i2c_perimage(probs, y):
    """Per-image (z-scored) metric vectors; bootstrap just averages them over images."""
    Z = (probs - probs.mean(0)) / (probs.std(0) + 1e-8)
    keep = y.sum(1) >= 1
    Z, Y = Z[keep], y[keep]
    n, L = Z.shape
    order = np.argsort(-Z, axis=1, kind="stable")
    rr = np.take_along_axis(Y, order, axis=1).astype(np.float64)   # relevance in ranked order
    m = Y.sum(1).astype(np.float64)
    cols = {}
    for k in KS_I2C:
        hits = rr[:, :k].sum(1)
        cols[f"P@{k}"] = hits / k
        cols[f"R@{k}"] = hits / m
    cum = np.cumsum(rr, axis=1)
    ranks = np.arange(1, L + 1)
    cols["mAP"] = (cum / ranks * rr).sum(1) / m                    # per-image average precision
    cum_neg = np.cumsum(1.0 - rr, axis=1)
    nneg = (L - m)
    neg_below = (nneg[:, None] - cum_neg)
    auc = (neg_below * rr).sum(1) / np.where((m * nneg) > 0, m * nneg, np.nan)
    cols["AUROC_img"] = auc                                        # nan where m==0 or m==L
    return cols, keep


def ci(vals):
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()
    t0 = time.time()
    rows_c2i, rep_c2i = [], {}
    rows_i2c, rep_i2c = [], {}

    for ds in args.datasets:
        df, _, y, labels, _ = common.load_dataset(ds)
        y = (y == 1).astype(np.int64)
        pid = patient_ids(ds, df)
        idx_sets = boot_index_sets(pid, args.B, args.seed)
        print(f"\n[{ds}] N={len(df)} patients={len(np.unique(pid))} B={args.B}", flush=True)

        S = {}
        for m, (rd, mode) in MODELS.items():
            pf = Path(EXP1) / rd / f"{ds}__{mode}__probs.npy"
            if not pf.exists():
                continue
            probs = np.load(pf).astype(np.float64)
            if probs.shape != y.shape:
                print(f"  [skip {m}] shape {probs.shape} != y {y.shape}", flush=True)
                continue
            S[m] = probs

        # concept->image
        c2i_b = {m: {me: [] for me in C2I_METRICS} for m in S}
        for b, idx in enumerate(idx_sets):
            for m in S:
                r = c2i_macro(S[m], y, idx)
                for me in C2I_METRICS:
                    c2i_b[m][me].append(r[me])
            if (b + 1) % 250 == 0:
                print(f"  c2i {b+1}/{args.B}", flush=True)
        for m in S:
            point = c2i_macro(S[m], y, np.arange(len(y)))
            row = {"model": m, "dataset": ds}
            for me in C2I_METRICS:
                lo, hi = ci(c2i_b[m][me])
                row[me] = point[me]; row[f"{me}_lo"] = lo; row[f"{me}_hi"] = hi
            rows_c2i.append(row); rep_c2i[(m, ds)] = c2i_b[m]

        # image->concept
        per = {m: i2c_perimage(S[m], y) for m in S}
        i2c_b = {m: {me: [] for me in I2C_METRICS} for m in S}
        for idx in idx_sets:
            for m in S:
                cols, keep = per[m]
                pos = np.cumsum(keep) - 1
                sel = idx[keep[idx]]
                kp = pos[sel]
                for me in I2C_METRICS:
                    i2c_b[m][me].append(float(np.nanmean(cols[me][kp])))
        for m in S:
            cols, _ = per[m]
            row = {"model": m, "dataset": ds}
            for me in I2C_METRICS:
                lo, hi = ci(i2c_b[m][me])
                row[me] = float(np.nanmean(cols[me])); row[f"{me}_lo"] = lo; row[f"{me}_hi"] = hi
            rows_i2c.append(row); rep_i2c[(m, ds)] = i2c_b[m]
        print(f"  [{ds}] done {time.time()-t0:.0f}s", flush=True)

    write_outputs("concept_to_image", "Concept->image retrieval", C2I_METRICS,
                  rows_c2i, rep_c2i, args)
    write_outputs("image_to_concept", "Image->concept annotation", I2C_METRICS,
                  rows_i2c, rep_i2c, args)
    print(f"\n[done] {time.time()-t0:.0f}s", flush=True)


def write_outputs(tag, title, metrics, rows, rep, args):
    flat = pd.DataFrame(rows)
    flat.to_csv(OUT / f"bootstrap_ci_{tag}.csv", index=False)
    sort_key = "pooled_R@10" if "pooled_R@10" in metrics else "AUROC_img"
    md = [f"# {title} — patient-clustered bootstrap 95% CI (B={args.B})\n",
          "Each cell = point [2.5%, 97.5%]. `*` after a baseline's value = ours beats it on "
          "that metric with a paired 95% CI excluding 0; `(w)` = baseline significantly beats "
          "ours. Datasets ordered as run.\n"]
    for ds in args.datasets:
        sub = flat[flat["dataset"] == ds]
        if sub.empty:
            continue
        md.append(f"\n## {ds}\n")
        md.append("| model | " + " | ".join(metrics) + " |")
        md.append("|" + "|".join(["---"] * (len(metrics) + 1)) + "|")
        for _, r in sub.sort_values(sort_key, ascending=False).iterrows():
            m = r["model"]; cells = []
            for me in metrics:
                c = f"{r[me]:.3f} [{r[me+'_lo']:.3f}, {r[me+'_hi']:.3f}]"
                if m != "ours":
                    d = np.asarray(rep[("ours", ds)][me]) - np.asarray(rep[(m, ds)][me])
                    dlo, dhi = ci(d)
                    if dlo > 0:
                        c += " *"
                    elif dhi < 0:
                        c += " (w)"
                cells.append(c)
            name = "**ours**" if m == "ours" else m
            md.append(f"| {name} | " + " | ".join(cells) + " |")
    (OUT / f"bootstrap_ci_{tag}.md").write_text("\n".join(md) + "\n")
    print(f"[wrote] {OUT}/bootstrap_ci_{tag}.{{csv,md}}", flush=True)


if __name__ == "__main__":
    main()

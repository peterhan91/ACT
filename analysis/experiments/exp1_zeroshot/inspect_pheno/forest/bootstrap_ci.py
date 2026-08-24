#!/usr/bin/env python3
"""
Per-phenotype zero-shot AUROC + 95% bootstrap CI — CheXzero-faithful.

Reproduces the confidence-interval procedure from CheXzero (Tiu et al.,
Nat. Biomed. Eng. 2022; Fig. 3), whose reference implementation is vendored at
  cxr_concept/external_sota/chexzero/code/CheXzero/eval.py  ::  bootstrap(), compute_cis()

Method (identical to CheXzero):
  * Non-parametric bootstrap over the TEST SET (the scan is the sampling unit).
  * np.random.seed(97); for i in range(n_samples): draw a resample of the SAME
    size WITH REPLACEMENT via sklearn.utils.resample(idx, replace=True,
    random_state=i); recompute per-label AUROC on that resample.
  * Because random_state=i is fixed per iteration, every model is scored on the
    SAME resampled indices -> the CIs are paired across models for free.
  * 95% CI = empirical percentiles of the n_samples bootstrap AUROCs, using
    CheXzero's exact index convention (confidence_level = 0.05):
        lower = sorted[int(0.025*N) - 1]   # ~2.5th pct
        upper = sorted[int(0.975*N) - 1]   # ~97.5th pct
    The reported centre is the bootstrap-distribution MEAN (matches CheXzero),
    and we also store the original-sample point AUROC for reference.

Inputs (all aligned byte-for-byte by inspect_pheno/build_assets.py):
  * each model's per-scan probs .npy  (n_volumes x n_model_labels)
  * each model's results.json         (gives that probs file's column order)
  * y_221.npy                         (n_volumes x 221 labels; col order = phecodes.csv)
  * phecodes.csv                      (the 221 label names, in y_221 column order)
  * exp4_86_phenotypes.csv            (which phenotypes to score; the 86 applicable)

Output:
  forest/ci_<tag>.csv  — long format: one row per (phenotype, model) with
  phecode, n_pos, auc_point, boot_mean, ci_lower, ci_upper, n_boot.

Usage:
  python bootstrap_ci.py                       # default: ours_f2llm + merlin, n=1000
  python bootstrap_ci.py --n-boot 1000 --tag f2llm_vs_merlin
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

# ── paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent                 # inspect_pheno/forest
INSPECT = HERE.parent                                  # inspect_pheno
EXP1 = INSPECT.parent                                  # exp1_zeroshot
EXPERIMENTS = EXP1.parent                              # experiments/
# repo root = .../clip_3d_concepts (exp1_zeroshot is under experiments/)
ROOT = EXP1.parents[1]

PHECODES_CSV = INSPECT / "phecodes.csv"                # 221 labels, y_221 column order
Y_221 = INSPECT / "y_221.npy"                          # (n_vol, 221) int labels — CLUSTER-SIDE
PERLABEL_CSV = INSPECT / "per_label_auc_all221_sorted.csv"   # has n_pos per phecode
LIST_86 = EXPERIMENTS / "exp4_confounder_audit" / "exp4_86_phenotypes.csv"  # the applicable 86

# model registry: name -> (probs.npy, results.json with the matching "labels" order)
MODELS = {
    "ours_f2llm": dict(
        display="Ours (f2llm)",
        probs=ROOT / "outputs/v1/external/phenotype__test__f2llm__probs.npy",
        results=ROOT / "outputs/v1/external/phenotype__test__f2llm__results.json",
    ),
    "merlin": dict(
        display="Merlin",
        probs=EXP1 / "merlin/results/inspect_pheno__native__probs.npy",
        results=EXP1 / "merlin/results/inspect_pheno__native__results.json",
    ),
    "ctclip": dict(
        display="CT-CLIP (CT-RATE)",
        probs=EXP1 / "ctclip/results/inspect_pheno__native__probs.npy",
        results=EXP1 / "ctclip/results/inspect_pheno__native__results.json",
    ),
}

SEED = 97              # CheXzero's np.random.seed(97)
CONFIDENCE = 0.05      # -> 95% CI


def compute_cis(boot: np.ndarray, confidence_level: float = CONFIDENCE):
    """CheXzero compute_cis(), per-column. boot: (n_samples, n_labels).
    Returns (mean, lower, upper) arrays of length n_labels."""
    n = boot.shape[0]
    lower_index = int(confidence_level / 2 * n) - 1
    upper_index = int((1 - confidence_level / 2) * n) - 1
    out_mean, out_lo, out_hi = [], [], []
    for j in range(boot.shape[1]):
        s = np.sort(boot[:, j][~np.isnan(boot[:, j])])
        if len(s) == 0:
            out_mean.append(np.nan); out_lo.append(np.nan); out_hi.append(np.nan); continue
        # guard the (rare) case where NaNs shrank the column below the index
        lo = s[min(max(lower_index, 0), len(s) - 1)]
        hi = s[min(max(upper_index, 0), len(s) - 1)]
        out_mean.append(float(s.mean())); out_lo.append(float(lo)); out_hi.append(float(hi))
    return np.array(out_mean), np.array(out_lo), np.array(out_hi)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC, or NaN if a (resampled) column is single-class."""
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, p)


def bootstrap_aucs(scores: np.ndarray, labels: np.ndarray, n_samples: int) -> np.ndarray:
    """(n_samples, n_labels) bootstrap AUROCs. scores/labels: (n_vol, n_labels),
    columns already aligned. Same resample indices across all labels per iter."""
    np.random.seed(SEED)
    n_vol, n_lab = scores.shape
    idx0 = np.arange(n_vol)
    boot = np.empty((n_samples, n_lab), dtype=float)
    for i in range(n_samples):
        sample = resample(idx0, replace=True, random_state=i)   # CheXzero: random_state=i
        ys, ps = labels[sample], scores[sample]
        for j in range(n_lab):
            boot[i, j] = _safe_auc(ys[:, j], ps[:, j])
    return boot


def load_aligned(model: dict, names: list[str]) -> np.ndarray:
    """Return that model's probs columns reordered to `names` (n_vol, len(names))."""
    d = json.loads(Path(model["results"]).read_text())
    col = {str(l).strip(): k for k, l in enumerate(d["labels"])}
    probs = np.load(model["probs"])
    miss = [n for n in names if n not in col]
    if miss:
        raise SystemExit(f"{model['display']}: {len(miss)} of {len(names)} phenotypes "
                         f"absent from its results.json labels, e.g. {miss[:3]}")
    return probs[:, [col[n] for n in names]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)        # CheXzero default
    ap.add_argument("--models", nargs="+", default=["ours_f2llm", "ctclip"])  # +merlin available
    ap.add_argument("--tag", default="f2llm")
    args = ap.parse_args()

    if not Y_221.exists():
        raise SystemExit(
            f"\nLABELS MISSING: {Y_221}\n"
            "y_221.npy (the 2612x221 INSPECT phenotype label matrix) is cluster-side and\n"
            "not synced (no PHI, but not in git/HF). scp it to compute CIs, e.g.:\n"
            "  scp <cluster>:/path/to/ACT/analysis/experiments/"
            "exp1_zeroshot/inspect_pheno/y_221.npy \\\n"
            f"      {Y_221}\n"
            "Then re-run. (plot_forest.py can still preview point AUROCs without it.)\n")

    # the 86 applicable phenotypes, in the file's order (we re-sort at plot time)
    names = pd.read_csv(LIST_86)["phenotype"].str.strip().tolist()
    # canonical label index (y_221 column order) + n_pos
    phe = pd.read_csv(PHECODES_CSV, dtype=str); phe["phecode_str"] = phe["phecode_str"].str.strip()
    lab_col = {n: k for k, n in enumerate(phe["phecode_str"])}
    y_all = np.load(Y_221)
    assert y_all.shape[1] == len(phe), f"y_221 has {y_all.shape[1]} cols, phecodes has {len(phe)}"
    y = y_all[:, [lab_col[n] for n in names]].astype(int)        # (n_vol, 86)
    pl = pd.read_csv(PERLABEL_CSV); pl = pl.rename(columns={pl.columns[0]: "name"})
    npos_by_name = {str(k).strip(): v for k, v in zip(pl["name"], pl["n_pos"])}
    phecode_of = dict(zip(phe["phecode_str"], phe["phecode"]))

    rows = []
    for m in args.models:
        scores = load_aligned(MODELS[m], names)                  # (n_vol, 86)
        # sanity: full-sample AUROC must match the model's published results.json
        pt = np.array([_safe_auc(y[:, j], scores[:, j]) for j in range(len(names))])
        _d = json.loads(Path(MODELS[m]["results"]).read_text())
        pub = _d["per_label_auc"]
        pub = pub if isinstance(pub, dict) else dict(zip(_d["labels"], pub))
        pub = {str(k).strip(): v for k, v in pub.items()}
        dev = np.nanmax([abs(pt[j] - pub[names[j]]) for j in range(len(names))])
        print(f"[{m}] point-AUROC vs results.json max |Δ| = {dev:.4g} "
              f"({'OK' if dev < 1e-6 else 'MISALIGNED — check row order!'})")
        boot = bootstrap_aucs(scores, y, args.n_boot)
        bmean, blo, bhi = compute_cis(boot)
        for j, nm in enumerate(names):
            rows.append(dict(phenotype=nm, phecode=phecode_of.get(nm, ""),
                             n_pos=int(npos_by_name.get(nm, 0) or 0),
                             model=m, display=MODELS[m]["display"],
                             auc_point=pt[j], boot_mean=bmean[j],
                             ci_lower=blo[j], ci_upper=bhi[j], n_boot=args.n_boot))

    out = HERE / f"ci_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(rows)} rows = {len(names)} phenotypes x {len(args.models)} models)")


if __name__ == "__main__":
    main()

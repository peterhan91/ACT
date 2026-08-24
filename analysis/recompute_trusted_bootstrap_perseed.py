#!/usr/bin/env python3
"""Recompute the *per-seed* trusted-probe concept importances (for scatter plots).

The committed bootstrap artifact
    outputs/<tag>/external/trusted_concept_probe_f2llm_lbfgs/
        trusted_f2llm_bootstrap_concept_importance.json
only stored the *summary* of the 20 train-resample seeds (importance_boot_mean /
_std / sign_consistency) — not the 20 raw per-seed importances. The middle panel
of experiments/exp4_confounder_audit/plot_top_gains.py wants those raw values so it
can draw per-seed scatter (like the left panel). This script regenerates them.

It must run where the INSPECT phenotype TRAIN labels live (the cluster), because
the probe is refit on bootstrap resamples of the train set — the train labels
(per_ct parquet) are patient-derived and never synced to HF/Mac.

Method (identical to trusted_concept_probe.py, whose helpers we import so the
feature pipeline matches bit-for-bit):
  x_train[label] = L2norm( (img_norm @ clip_emb_sel.T) @ f2llm_emb_sel )   # (N, 5120)
  for seed in 1..n_seeds:
      idx = default_rng(seed).integers(0, N, N)          # train bootstrap resample
      xb, yb = x_train[idx], y_train[idx]
      z-score xb with ITS OWN per-resample mean/sd        (sd_b = xb.std(0)+1e-8)
      w_z_b = LogisticRegression(C=1e6, solver='lbfgs', max_iter=5000).fit(z(xb), yb).coef_
      importance_b[k] = f2llm_emb_sel[k] . (w_z_b / sd_b)  # back-projection onto concept k
The importance formula is verified against the stored importance_point (full-train
fit) before any per-seed value is written; mismatch aborts so a pipeline drift is
never silently plotted.

Output: trusted_f2llm_bootstrap_concept_importance_perseed.json  — the SAME schema
as the input bootstrap json, but every positive concept also carries a `per_seed`
list of length n_seeds. plot_top_gains.py prefers this file automatically.

Usage (on the cluster, after `conda activate` the probe env):
    python recompute_trusted_bootstrap_perseed.py            # the 8 figure phenotypes
    python recompute_trusted_bootstrap_perseed.py --all      # all 87 labels
    python recompute_trusted_bootstrap_perseed.py --labels "Shock" "Pneumococcal pneumonia"
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CLIP3D_RUN_TAG", "v1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from clear3d.paths import CACHE, EXTERNAL_DIR, PHENOTYPE_LABELS, PHENOTYPE_MANIFEST, ROOT
import trusted_concept_probe as tcp

TRUSTED_DIR = EXTERNAL_DIR / "trusted_concept_probe_f2llm_lbfgs"
BOOT_JSON = TRUSTED_DIR / "trusted_f2llm_bootstrap_concept_importance.json"
OUT_JSON = TRUSTED_DIR / "trusted_f2llm_bootstrap_concept_importance_perseed.json"
REFINE = ROOT / "experiments/exp4_confounder_audit/results/f2llm_refinement_all86_concepts.json"

# bootstrap recipe (from trusted_f2llm_bootstrap_meta.json)
C = 1e6
MAX_ITER = 5000
BALANCED = False
EPS = 1e-8


def importance_from_fit(emb_sel: np.ndarray, coef: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Per-concept importance = normalized concept embedding . (w_z / sd)."""
    return emb_sel.astype(np.float64) @ (coef.astype(np.float64) / sd.astype(np.float64))


def fit_coef_sd(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score x with its own train stats, fit LBFGS, return (coef_z, sd)."""
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True) + EPS
    xz = (x - mu) / sd
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=MAX_ITER,
                               class_weight="balanced" if BALANCED else None)
    model.fit(xz, y.astype(int))
    return np.asarray(model.coef_).reshape(-1), sd.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="all 87 labels (default: the 8 figure phenotypes)")
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--trusted_dir", default=str(ROOT / "outputs" / os.environ["CLIP3D_RUN_TAG"] / "audit" / "trusted_concept_spaces"))
    ap.add_argument("--audit_json", default=str(ROOT / "outputs" / os.environ["CLIP3D_RUN_TAG"] / "audit" / "phenotype__linear_f2llm__concept_importance.json"))
    # Cluster-only PHI by default; override to local copies so this can run off-cluster.
    ap.add_argument("--labels_parquet", default=str(PHENOTYPE_LABELS))
    ap.add_argument("--manifest_parquet", default=str(PHENOTYPE_MANIFEST))
    ap.add_argument("--min_test_positives", type=int, default=50)
    # 0 = keep ALL trusted concepts per label (matches _audit_trusted_bootstrap.py,
    # which does NOT cap — e.g. Shock uses 781). A 512 cap would change the 5120-d
    # projection and fail the importance_point drift gate, so keep 0.
    ap.add_argument("--max_concepts", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=0.02, help="abort if point-fit importance drifts > rtol from stored importance_point")
    args = ap.parse_args()

    boot = json.loads(BOOT_JSON.read_text())

    if args.labels:
        targets = list(args.labels)
    elif args.all:
        targets = [k for k, v in boot.items() if "trusted" in v]
    else:
        refine = json.loads(REFINE.read_text())
        targets = [r["phenotype"] for r in sorted(refine, key=lambda z: -z["auc_gain"])[:8]]
    print(f"targets ({len(targets)}): {targets}")

    # ---- replicate trusted_concept_probe.main() data setup (train split + labels) ----
    labels_df = pd.read_parquet(args.labels_parquet)
    manifest = pd.read_parquet(args.manifest_parquet)
    phecode_to_str = tcp.load_phecode_name_map(EXTERNAL_DIR)
    lookup = tcp.build_inspect_lookup()

    test_manifest = manifest[(manifest["split"] == "test") & (manifest["visit_occurrence_id"].notna())]
    test_vols = [v for v in test_manifest["VolumeName"].values if v in lookup["vol_to_row"]]
    prev = labels_df.loc[test_vols].sum(axis=0)
    keep_phecodes = sorted(prev[prev >= args.min_test_positives].index.tolist())
    keep_names = [phecode_to_str.get(p, p) for p in keep_phecodes]
    name_to_i = {n: i for i, n in enumerate(keep_names)}

    train = tcp.gather_for_split(manifest[manifest["split"] == "train"], labels_df, keep_phecodes, lookup)
    train["img"] = tcp.normalize_rows(train["img"])
    img_train = train["img"]
    print(f"train volumes: {len(train['y'])}")

    bank = tcp.IndexedConceptBanks({
        "clip": ROOT / f"concept_bank.clip_text_emb.{os.environ['CLIP3D_RUN_TAG']}.npz",
        "f2llm": ROOT / "concept_bank.f2llm_emb.npz",
    })
    summary_df = pd.read_csv(Path(args.trusted_dir) / "summary.csv")
    summary_by_label = {str(r["label"]): dict(r) for _, r in summary_df.iterrows()}

    out = {}
    for ph in targets:
        if ph not in name_to_i:
            print(f"[skip] {ph}: not in filtered label set")
            continue
        li = name_to_i[ph]
        cset = tcp.trusted_set(ph, summary_by_label, Path(args.trusted_dir),
                               max_count=args.max_concepts, direct_only=False)
        if not cset.indices:
            print(f"[skip] {ph}: no trusted concepts")
            continue
        clip_sel = bank.select_embeddings(cset.indices, "clip")
        f2_sel = bank.select_embeddings(cset.indices, "f2llm")        # (K,5120) normalized = importance basis
        x_train = tcp.llm_concept_projection(img_train, clip_sel, f2_sel)  # (N,5120)
        y_train = train["y"][:, li].astype(np.float32)
        N = len(y_train)
        ci_to_k = {int(c): k for k, c in enumerate(cset.indices)}

        # point fit (full train) -> verify against stored importance_point
        coef0, sd0 = fit_coef_sd(x_train, y_train)
        imp0 = importance_from_fit(f2_sel, coef0, sd0)
        stored = boot[ph]["trusted"]["positive"]
        drift = []
        for c in stored:
            k = ci_to_k.get(int(c["concept_index"]))
            if k is not None and c["importance_point"]:
                drift.append(abs(imp0[k] - c["importance_point"]) / (abs(c["importance_point"]) + 1e-9))
        max_drift = max(drift) if drift else float("nan")
        print(f"  {ph:42s} N={N} K={len(cset.indices)}  point-fit max rel drift={max_drift:.4f}")
        if drift and max_drift > args.rtol:
            raise SystemExit(f"ABORT: {ph} point-fit importance drifts {max_drift:.3f} > rtol {args.rtol} "
                             f"-> feature/label pipeline mismatch; do not trust per-seed.")

        # per-seed bootstrap refits
        per_seed = np.zeros((args.n_seeds, len(cset.indices)), np.float64)
        for s in range(1, args.n_seeds + 1):
            idx = np.random.default_rng(s).integers(0, N, N)
            coef_b, sd_b = fit_coef_sd(x_train[idx], y_train[idx])
            per_seed[s - 1] = importance_from_fit(f2_sel, coef_b, sd_b)

        # attach per_seed to the stored positives; cross-check boot mean/std
        node = json.loads(json.dumps(boot[ph]))   # deep copy
        for c in node["trusted"]["positive"]:
            k = ci_to_k.get(int(c["concept_index"]))
            if k is None:
                continue
            vals = per_seed[:, k]
            c["per_seed"] = [float(v) for v in vals]
            c["_recompute_boot_mean"] = float(vals.mean())
            c["_recompute_boot_std"] = float(vals.std(ddof=1))
        out[ph] = node

    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT_JSON}  ({len(out)} phenotypes, n_seeds={args.n_seeds})")
    print("Pull to Mac, then re-run plot_top_gains.py — scatter will render automatically.")
    print(f"  python tools/hf_sync.py push refinement   # if you want it on HF")


if __name__ == "__main__":
    main()

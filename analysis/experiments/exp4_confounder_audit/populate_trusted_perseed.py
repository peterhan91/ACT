#!/usr/bin/env python3
"""Populate `per_seed` in the trusted-probe bootstrap importance JSON from the NPZ.

The HF artifact `trusted_f2llm_bootstrap_concept_importance_perseed.json` ships with
EMPTY `per_seed` lists (it is a stub copy of the summary JSON). The actual 20-seed
train-bootstrap importances live in the companion array bundle:

    trusted_f2llm_bootstrap_perseed_coefs.npz   (231 MB)
        <slug>__trusted__importance   (21, K)   row0 = full-train point fit,
                                                 rows1..20 = bootstrap seeds 1..20
                                                 importance = L2norm(llm_emb)·(w_z/sd)
        <slug>__trusted__indices      (K,)       concept_index per column
    trusted_f2llm_bootstrap_perseed_index.json   phenotype-name -> {slug, methods}

This fills each positive concept's `per_seed` (the 20 bootstrap rows) by matching
its `concept_index` into the NPZ `indices`, verifying importance[0] ≈ stored
importance_point (drift gate) before writing. plot_top_gains.py then renders
per-seed scatter on the trusted (blue) panel, symmetric with the naive (red) one.

This is an off-cluster completion of recompute_trusted_bootstrap_perseed.py: it
reuses the already-computed NPZ instead of refitting (which needs cluster-only PHI
train labels), so it runs on the Mac.

    python experiments/exp4_confounder_audit/populate_trusted_perseed.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TDIR = ROOT / "outputs/v1/external/trusted_concept_probe_f2llm_lbfgs"
SUMMARY = TDIR / "trusted_f2llm_bootstrap_concept_importance.json"
INDEX = TDIR / "trusted_f2llm_bootstrap_perseed_index.json"
NPZ = TDIR / "trusted_f2llm_bootstrap_perseed_coefs.npz"
OUT = TDIR / "trusted_f2llm_bootstrap_concept_importance_perseed.json"

RTOL = 0.02  # abort a concept's per_seed if NPZ point-fit drifts > 2% from stored point


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    index = json.loads(INDEX.read_text())
    z = np.load(NPZ, allow_pickle=True)

    out = json.loads(json.dumps(summary))  # deep copy; keep schema identical
    n_ph = n_concept = n_filled = n_drift = n_missing = 0

    for ph, node in out.items():
        meta = index.get(ph)
        if not meta or "trusted" not in node:
            continue
        slug = meta["slug"]
        ikey, ckey = f"{slug}__trusted__indices", f"{slug}__trusted__importance"
        if ikey not in z.files or ckey not in z.files:
            continue
        indices = z[ikey].astype(np.int64)
        imp = z[ckey].astype(np.float64)            # (21, K): row0 point, 1..20 seeds
        ci_to_k = {int(c): k for k, c in enumerate(indices)}
        n_ph += 1
        for pol in ("positive", "negative"):
            for c in node["trusted"].get(pol, []):
                n_concept += 1
                k = ci_to_k.get(int(c.get("concept_index", -1)))
                if k is None:
                    n_missing += 1
                    continue
                point_npz = float(imp[0, k])
                stored = float(c.get("importance_point", 0.0))
                if abs(stored) > 1e-9 and abs(point_npz - stored) / abs(stored) > RTOL:
                    n_drift += 1
                    continue
                c["per_seed"] = [float(v) for v in imp[1:, k]]  # 20 bootstrap seeds
                n_filled += 1

    OUT.write_text(json.dumps(out, indent=1))
    print(f"phenotypes matched : {n_ph}")
    print(f"concepts seen      : {n_concept}")
    print(f"per_seed filled    : {n_filled}")
    print(f"skipped (drift>2%) : {n_drift}")
    print(f"skipped (no index) : {n_missing}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

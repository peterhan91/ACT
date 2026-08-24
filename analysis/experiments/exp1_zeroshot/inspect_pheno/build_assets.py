#!/usr/bin/env python3
"""Build the INSPECT-phenotype baseline-eval assets, aligned BYTE-FOR-BYTE to the
2,612 test volumes + 221 phecodes that ours (exp_phenotype_ct.py) scored, so every
baseline's per-phecode AUROC is directly comparable to ours.

Writes (in this dir):
  manifest.csv   VolumeName, Path (native NIfTI), h5_idx (row index)
  y_221.npy      (2612, 221) int8 label matrix, column order = phecodes.csv
  phecodes.csv   phecode, phecode_str   (221 rows, sorted phecode order)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PHENO = Path("/path/to/phenotype_labels/per_ct")
CACHE = Path("/path/to/ACT/analysis/outputs/v1/cache")   # ours-cache (volume_index)
NIFTI_ROOT = Path("/path/to/data/Inspect_v2.0/"
                  "inspectamultimodaldatasetforpulmonaryembolismdiagnosisandprog-3/full/CTPA")

labels = pd.read_parquet(PHENO / "per_ct_labels_visit_only.parquet")   # 22461 x 1692
manifest = pd.read_parquet(PHENO / "manifest.parquet")
phen = pd.read_csv(PHENO / "phenotypes.csv", dtype=str)
phen["phecode"] = phen["phecode"].str.strip(); phen["phecode_str"] = phen["phecode_str"].str.strip()
name = dict(zip(phen["phecode"], phen["phecode_str"]))

# ours-cache volume set (union of the 3 inspect_* volume_index csvs)
cache = set()
for s in ["inspect_train", "inspect_valid", "inspect_test"]:
    cache |= set(pd.read_csv(CACHE / f"volume_index.{s}.csv")["VolumeName"])
li = set(labels.index)

# --- replicate ours' keep_phecodes (>=50 positives in test, with-visit, in cache) ---
tm = manifest[(manifest.split == "test") & (manifest.visit_occurrence_id.notna())]
test_vols0 = [v for v in tm.VolumeName if v in cache]
prev = labels.loc[test_vols0].sum(0)
keep = sorted(prev[prev >= 50].index.tolist())
assert len(keep) == 221, len(keep)

# --- replicate ours' exact test-volume order (gather_for_split) ---
vols = [v for v in tm.VolumeName if v in cache and v in li]
assert len(vols) == 2612, len(vols)
y = labels.loc[vols][keep].values.astype(np.int8)

# --- NIfTI path per volume + existence check ---
paths = [str(NIFTI_ROOT / v) for v in vols]
missing = [p for p in paths if not Path(p).exists()]
print(f"vols={len(vols)}  phecodes={len(keep)}  y={y.shape}  missing_nifti={len(missing)}")
if missing:
    print("  first missing:", missing[:3])
    raise SystemExit("some INSPECT NIfTIs are absent — fix path mapping before launch")

# --- write assets ---
pd.DataFrame({"VolumeName": vols, "Path": paths,
             "h5_idx": np.arange(len(vols), dtype=np.int64)}).to_csv(HERE / "manifest.csv", index=False)
np.save(HERE / "y_221.npy", y)
pd.DataFrame({"phecode": keep, "phecode_str": [name.get(p, p) for p in keep]}
             ).to_csv(HERE / "phecodes.csv", index=False)
# also the level-1 (integer) subset mask for convenience
is_int = [("." not in p) for p in keep]
json.dump({"n_vols": len(vols), "n_phecodes": len(keep),
           "n_level1_integer": int(sum(is_int))},
          open(HERE / "_meta.json", "w"), indent=2)
print(f"wrote manifest.csv, y_221.npy ({y.shape}), phecodes.csv (221), level-1={sum(is_int)}")
print("prevalence sanity: pos rate min/mean/max =",
      round(y.mean(0).min(), 4), round(y.mean(0).mean(), 4), round(y.mean(0).max(), 4))

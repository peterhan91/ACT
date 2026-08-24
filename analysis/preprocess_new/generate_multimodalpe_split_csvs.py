#!/usr/bin/env python3
"""Generate MultimodalPE train/valid/test path CSVs using the official split
column in Labels.csv (Huang 2020 Stanford split: 1454 train / 193 val / 190 test).

Inputs:
  --paths_csv : master path CSV from multimodalpe_npy2nifti.py
                (filenames: <subdir>__<idx>.nii.gz)
  --labels_csv: Labels.csv shipped with the dataset
                (columns: idx, label, pe_type, split)
  --out_dir   : destination for the three split CSVs

Output filename convention matches CT-RATE/INSPECT/MERLIN:
    multimodalpe_train_paths.csv
    multimodalpe_valid_paths.csv
    multimodalpe_test_paths.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths_csv", required=True,
                    help="Master paths CSV produced by multimodalpe_npy2nifti.py")
    ap.add_argument("--labels_csv", required=True,
                    help="MultimodalPE Labels.csv (with `idx`, `label`, `pe_type`, `split`)")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write multimodalpe_{train,valid,test}_paths.csv")
    args = ap.parse_args()

    paths = pd.read_csv(args.paths_csv)
    paths["fname"] = paths["Path"].apply(lambda p: Path(p).name.replace(".nii.gz", ""))
    # filename is "<subdir>__<idx>"
    paths["idx"] = paths["fname"].apply(lambda n: int(n.split("__")[-1]))
    print(f"[load] {len(paths)} NIfTI volumes  (unique idx={paths.idx.nunique()})")

    labels = pd.read_csv(args.labels_csv)
    print(f"[load] {len(labels)} label rows  splits={labels['split'].value_counts().to_dict()}")

    # Inner-join on idx; report any orphan rows.
    merged = paths.merge(labels[["idx", "split"]], on="idx", how="inner")
    n_drop = len(labels) - len(merged)
    if n_drop:
        missing = set(labels["idx"]) - set(paths["idx"])
        print(f"[warn] {n_drop} label rows have no matching NIfTI; sample missing idx={list(missing)[:5]}")
    extra = set(paths["idx"]) - set(labels["idx"])
    if extra:
        print(f"[warn] {len(extra)} NIfTI files have no label row; sample idx={sorted(extra)[:5]}")

    # Map dataset's "val" -> our "valid" naming convention.
    split_map = {"train": "train", "val": "valid", "test": "test"}
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for src_split, dst_split in split_map.items():
        sub = merged[merged["split"] == src_split].sort_values("idx")
        out_csv = out_root / f"multimodalpe_{dst_split}_paths.csv"
        pd.DataFrame({"Path": sub["Path"].tolist()}).to_csv(out_csv, index=False)
        print(f"[write] {out_csv}  ({len(sub)} volumes)")


if __name__ == "__main__":
    main()

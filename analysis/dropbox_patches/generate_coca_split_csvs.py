#!/usr/bin/env python3
"""Generate COCA train/valid/test path CSVs from the master `_paths.csv`
written by dicom_series_to_nifti.py.

PATIENT-OVERLAP HANDLING
------------------------
COCA ships in two subsets: 787 gated patients + 214 nongated patients.
A direct check on the on-disk directory names shows that 211 of 214
nongated IDs collide numerically with gated IDs. We do NOT know whether
identical numeric IDs across subsets refer to the same person, so the
safe default is to treat them as the same patient and split on patient_id
alone (no subset stratification). This guarantees no cross-split leakage
even in the worst case.

Pass --by_subset to instead stratify by gated/nongated (keeps both
acquisition protocols proportionally represented in each split). Use only
if you have external evidence that gated and nongated ID spaces are
independent.

Default ratio: 80% train / 10% valid / 10% test, seed 42.

Output (matching CT-RATE/INSPECT/MERLIN convention):
    coca_train_paths.csv
    coca_valid_paths.csv
    coca_test_paths.csv

NIfTI filename layout produced by the converter (input_dir = .../cocacoronarycalciumandchestcts-2):
    Gated_release_final__patient__<pid>__<series_dirname>.nii.gz
    deidentified_nongated__<pid>.nii.gz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_subset_and_pid(nifti_basename: str) -> tuple[str, str]:
    """Returns (subset, patient_id). subset ∈ {"gated", "nongated", "other"}."""
    name = nifti_basename.replace(".nii.gz", "")
    parts = name.split("__")
    if parts[0].startswith("Gated_release_final") and len(parts) >= 3:
        return ("gated", parts[2])
    if parts[0].startswith("deidentified_nongated") and len(parts) >= 2:
        return ("nongated", parts[1])
    return ("other", parts[0])


def _split_pids(pids: list[str], rng: np.random.Generator,
                ratios: tuple[float, float, float]) -> tuple[set, set, set]:
    pids = sorted(pids)
    rng.shuffle(pids)
    n = len(pids)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    return (set(pids[:n_tr]),
            set(pids[n_tr:n_tr + n_va]),
            set(pids[n_tr + n_va:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths_csv", required=True,
                    help="Master paths CSV produced by dicom_series_to_nifti.py")
    ap.add_argument("--out_dir", required=True,
                    help="Where to write coca_{train,valid,test}_paths.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    metavar=("TRAIN", "VALID", "TEST"))
    ap.add_argument("--by_subset", action="store_true",
                    help="Stratify split by gated/nongated subset. Off by default "
                         "because ~99%% of nongated PIDs collide numerically with "
                         "gated PIDs and we don't know if they're the same patient.")
    args = ap.parse_args()

    df = pd.read_csv(args.paths_csv)
    if df.empty:
        sys.exit(
            f"[split] {args.paths_csv} is empty — upstream DICOM->NIfTI step produced "
            f"0 NIfTI files. Inspect the conversion log / _failures.txt before continuing."
        )
    df["subset"], df["pid"] = zip(*df["Path"].apply(lambda p: parse_subset_and_pid(Path(p).name)))
    print(f"[load] {len(df)} NIfTI series  ({df['subset'].value_counts().to_dict()})")

    rng = np.random.default_rng(args.seed)
    splits: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    ratios = tuple(args.ratios)

    if args.by_subset:
        print("[split] mode: stratified by gated/nongated subset")
        for subset, sub in df.groupby("subset"):
            uniq_pids = sub["pid"].unique().tolist()
            train_pids, valid_pids, test_pids = _split_pids(uniq_pids, rng, ratios)
            print(f"  {subset:>9}: {len(uniq_pids)} pids → train={len(train_pids)}  valid={len(valid_pids)}  test={len(test_pids)}")
            for split, ids in (("train", train_pids), ("valid", valid_pids), ("test", test_pids)):
                splits[split].extend(sub[sub["pid"].isin(ids)]["Path"].tolist())
    else:
        print("[split] mode: unified by patient_id (gated and nongated share namespace)")
        uniq_pids = df["pid"].unique().tolist()
        train_pids, valid_pids, test_pids = _split_pids(uniq_pids, rng, ratios)
        print(f"  {len(uniq_pids)} unique pids → train={len(train_pids)}  valid={len(valid_pids)}  test={len(test_pids)}")
        for split, ids in (("train", train_pids), ("valid", valid_pids), ("test", test_pids)):
            splits[split].extend(df[df["pid"].isin(ids)]["Path"].tolist())

    # --- final hard assertion: no patient appears in more than one split ----
    split_pid_sets = {}
    for split, paths in splits.items():
        sub = df[df["Path"].isin(paths)]
        # collision key matches the chosen split mode
        if args.by_subset:
            split_pid_sets[split] = set(zip(sub["subset"], sub["pid"]))
        else:
            split_pid_sets[split] = set(sub["pid"])
    overlaps = []
    for a in ("train", "valid", "test"):
        for b in ("train", "valid", "test"):
            if a < b:
                inter = split_pid_sets[a] & split_pid_sets[b]
                if inter:
                    overlaps.append((a, b, len(inter), sorted(inter)[:5]))
    if overlaps:
        for a, b, n, samples in overlaps:
            print(f"[FATAL] patient overlap {a} ↔ {b}: {n} ids; samples={samples}")
        raise SystemExit(2)
    print("[check] OK — no patient_id appears in more than one split.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, paths in splits.items():
        paths = sorted(paths)
        out_csv = out / f"coca_{split}_paths.csv"
        pd.DataFrame({"Path": paths}).to_csv(out_csv, index=False)
        print(f"[write] {out_csv}  ({len(paths)} series)")


if __name__ == "__main__":
    main()

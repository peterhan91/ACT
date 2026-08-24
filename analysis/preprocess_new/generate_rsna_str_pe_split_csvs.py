#!/usr/bin/env python3
"""Study-wise train/valid/test split for the RSNA-STR PE Detection dataset.

Why study-wise (not patient-wise):

The kaggle release strips PatientID from every DICOM (verified empirically:
0/30 sampled headers carry a PatientID tag) AND train.csv exposes no patient
column. There is therefore no on-disk way to recover patient identity.

Per Colak et al., Radiology AI 2021 (10.1148/ryai.2021200254):

    "Finally, the final dataset was evaluated to ensure only one study per
     patient from a site; if there were multiple studies per patient, any
     additional studies from the patient were excluded (n = 129)."

That guarantees **within-site** uniqueness, so splitting on StudyInstanceUID
is equivalent to patient-wise WITHIN A SITE. The paper does **not** rule out
the same patient appearing across multiple of the 5 source sites, and since
the data has no Institution tag either (verified empirically) we cannot
detect such cross-site duplication. Every public benchmark on this dataset
inherits the same limitation.

Operationally:
  - Split key   : StudyInstanceUID
  - Ratios      : 80% / 10% / 10%, seed 42 (matches rsna2023 convention)
  - Hard check  : no StudyInstanceUID appears in more than one split
  - Labels      : aggregated from slice-level kaggle train.csv to study-level
                  (pe_present_on_study = any(pe_present_on_image), the other
                  13 binary columns are already study-constant → take .first()).
  - Test source : kaggle's labeled train/ tree only; kaggle's unlabeled
                  competition test/ DICOMs are not used.

Outputs (in --out_dir):
    rsna_str_pe_train_paths.csv     (one column: Path)
    rsna_str_pe_valid_paths.csv
    rsna_str_pe_test_paths.csv
    rsna_str_pe_train_labels.csv    (StudyInstanceUID, SeriesInstanceUID,
                                      Path, plus 14 study-level binary labels)
    rsna_str_pe_valid_labels.csv
    rsna_str_pe_test_labels.csv

NIfTI filename layout produced by dicom_series_to_nifti.py for
--input_dir=.../rsna-str-pulmonary-embolism-detection/train:
    <StudyInstanceUID>__<SeriesInstanceUID>.nii.gz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Study-level binary label columns in kaggle train.csv (all 1/0 across slices
# of the same study). pe_present_on_image is slice-level → handled separately
# and renamed to pe_present_on_study in the aggregation.
STUDY_LEVEL_BINARY = [
    "negative_exam_for_pe",
    "indeterminate",
    "rv_lv_ratio_gte_1",
    "rv_lv_ratio_lt_1",
    "leftsided_pe",
    "rightsided_pe",
    "central_pe",
    "acute_and_chronic_pe",
    "chronic_pe",
    "qa_motion",
    "qa_contrast",
    "flow_artifact",
    "true_filling_defect_not_pe",
]


def aggregate_study_level(slice_labels: pd.DataFrame) -> pd.DataFrame:
    """One row per StudyInstanceUID. pe_present_on_study = any pe_present_on_image."""
    if "pe_present_on_image" not in slice_labels.columns:
        sys.exit("train.csv is missing 'pe_present_on_image' column.")
    present = [c for c in STUDY_LEVEL_BINARY if c in slice_labels.columns]
    agg = {c: "first" for c in present}
    agg["pe_present_on_image"] = "max"  # → pe_present_on_study
    out = (
        slice_labels.groupby("StudyInstanceUID")
        .agg(agg)
        .rename(columns={"pe_present_on_image": "pe_present_on_study"})
        .reset_index()
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paths_csv", required=True,
        help="Master NIfTI paths CSV produced by dicom_series_to_nifti.py for the train/ dir",
    )
    ap.add_argument(
        "--labels_csv",
        default="/path/to/data/competitions/rsna-str-pulmonary-embolism-detection/train.csv",
        help="Kaggle train.csv (slice-level labels)",
    )
    ap.add_argument(
        "--out_dir", required=True,
        help="Where to write rsna_str_pe_{train,valid,test}_{paths,labels}.csv",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    metavar=("TRAIN", "VALID", "TEST"))
    args = ap.parse_args()

    # ------- load -------
    paths_df = pd.read_csv(args.paths_csv)
    if "Path" not in paths_df.columns:
        sys.exit(f"{args.paths_csv} missing 'Path' column")
    paths_df["fname"] = paths_df["Path"].apply(lambda p: Path(p).name)
    parts = paths_df["fname"].str.removesuffix(".nii.gz").str.split("__", n=1, expand=True)
    paths_df["StudyInstanceUID"] = parts[0]
    paths_df["SeriesInstanceUID"] = parts[1]
    print(f"[load] {len(paths_df)} NIfTI series across "
          f"{paths_df['StudyInstanceUID'].nunique()} unique studies")

    labels_slice = pd.read_csv(args.labels_csv)
    labels_study = aggregate_study_level(labels_slice)
    pe_prev = 100.0 * labels_study["pe_present_on_study"].mean()
    print(f"[load] aggregated kaggle labels to study-level: {len(labels_study)} studies; "
          f"pe_present_on_study prevalence {pe_prev:.2f}%")

    # ------- join -------
    merged = paths_df.merge(labels_study, on="StudyInstanceUID", how="inner")
    dropped_paths = len(paths_df) - len(merged)
    if dropped_paths:
        print(f"[warn] dropping {dropped_paths} NIfTI series with no matching label row")
    dropped_labels = len(labels_study) - merged["StudyInstanceUID"].nunique()
    if dropped_labels:
        print(f"[warn] {dropped_labels} labeled studies have no NIfTI on disk "
              f"(probably failed DICOM→NIfTI conversion)")

    # ------- study-wise shuffle + split -------
    rng = np.random.default_rng(args.seed)
    studies = sorted(merged["StudyInstanceUID"].unique())
    rng.shuffle(studies)

    n = len(studies)
    r_train, r_valid, _r_test = args.ratios
    n_tr = int(round(n * r_train))
    n_va = int(round(n * r_valid))
    train_studies = set(studies[:n_tr])
    valid_studies = set(studies[n_tr:n_tr + n_va])
    test_studies = set(studies[n_tr + n_va:])
    print(f"[split] {n} studies → train={len(train_studies)}  "
          f"valid={len(valid_studies)}  test={len(test_studies)}")

    # StudyInstanceUID-overlap hard check.
    name_to_set = {"train": train_studies, "valid": valid_studies, "test": test_studies}
    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    bad = [(a, b, name_to_set[a] & name_to_set[b]) for a, b in pairs
           if (name_to_set[a] & name_to_set[b])]
    if bad:
        for a, b, inter in bad:
            print(f"[FATAL] StudyInstanceUID overlap {a} ↔ {b}: {len(inter)} ids; "
                  f"samples={sorted(inter)[:5]}")
        raise SystemExit(2)
    print("[check] OK — no StudyInstanceUID appears in more than one split.")

    # ------- write -------
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label_cols = ["pe_present_on_study"] + [c for c in STUDY_LEVEL_BINARY if c in merged.columns]
    keep_cols = ["StudyInstanceUID", "SeriesInstanceUID", "Path"] + label_cols

    for split, sids in (("train", train_studies), ("valid", valid_studies),
                        ("test", test_studies)):
        sub = (merged[merged["StudyInstanceUID"].isin(sids)]
               .sort_values(["StudyInstanceUID", "SeriesInstanceUID"])
               .reset_index(drop=True))
        path_csv = out / f"rsna_str_pe_{split}_paths.csv"
        lab_csv = out / f"rsna_str_pe_{split}_labels.csv"
        pd.DataFrame({"Path": sub["Path"].tolist()}).to_csv(path_csv, index=False)
        sub[keep_cols].to_csv(lab_csv, index=False)
        prev = {c: float(sub[c].mean()) for c in label_cols}
        print(f"[write] {split}: {len(sub)} series, "
              f"{sub['StudyInstanceUID'].nunique()} studies → "
              f"{path_csv.name}, {lab_csv.name}")
        for c, v in prev.items():
            print(f"    {c:<28s}  prevalence {100*v:5.2f}%")


if __name__ == "__main__":
    main()

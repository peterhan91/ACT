#!/usr/bin/env python3
"""Per-labelset 80/20 stratified split manifests for LUNG1 + RADIO.

Reads the TCIA clinical CSVs, derives binary / multi-label targets per labelset,
drops rows with missing or ambiguous labels, writes one CSV per labelset with
columns (PatientID, <label cols>, split). Patient-level stratified on the
primary positive class, seed=0, 80/20.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
TEST_FRAC = 0.20


def stratified_split(y, test_frac, seed):
    """Per-class shuffle then take test_frac of each class. Returns test indices."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    test_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_te = max(1, int(round(len(idx) * test_frac)))
        test_idx.append(idx[:n_te])
    return np.concatenate(test_idx)

LUNG1_CSV_DEFAULT = (
    "/path/to/data/tcia/nsclc_radiomics/clinical/"
    "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
)
RADIO_CSV_DEFAULT = (
    "/path/to/data/tcia/nsclc_radiogenomics/clinical/"
    "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv"
)


# ---------- LUNG1 ----------------------------------------------------------
def lung1_os2yr(df):
    pos = (df["deadstatus.event"] == 1) & (df["Survival.time"] <= 730)
    neg = ((df["deadstatus.event"] == 1) & (df["Survival.time"] > 730)) | \
          ((df["deadstatus.event"] == 0) & (df["Survival.time"] >= 730))
    keep = pos | neg
    out = df.loc[keep, ["PatientID"]].copy()
    out["os2yr"] = pos[keep].astype(int).values
    return out, "os2yr"


def lung1_histology(df):
    keep = df["Histology"].notna()
    out = df.loc[keep, ["PatientID"]].copy()
    h = df.loc[keep, "Histology"]
    for src, name in [("adenocarcinoma", "adenocarcinoma"),
                      ("squamous cell carcinoma", "squamous_cell_carcinoma"),
                      ("large cell", "large_cell"),
                      ("nos", "nos")]:
        out[name] = (h == src).astype(int).values
    return out, "adenocarcinoma"  # rarest class


def lung1_stage_advanced(df):
    keep = df["Overall.Stage"].notna()
    out = df.loc[keep, ["PatientID"]].copy()
    out["stage_advanced"] = (
        df.loc[keep, "Overall.Stage"].str.startswith("III").astype(int).values
    )
    return out, "stage_advanced"


# ---------- RADIO ----------------------------------------------------------
def _radio_binary(df, src, name, pos_vals, neg_vals):
    is_pos = df[src].isin(pos_vals)
    is_neg = df[src].isin(neg_vals)
    keep = is_pos | is_neg
    out = df.loc[keep, ["Case ID"]].rename(columns={"Case ID": "PatientID"}).copy()
    out[name] = is_pos[keep].astype(int).values
    return out, name


def radio_egfr(df):
    return _radio_binary(df, "EGFR mutation status", "egfr",
                         {"Mutant"}, {"Wildtype"})


def radio_kras(df):
    return _radio_binary(df, "KRAS mutation status", "kras",
                         {"Mutant"}, {"Wildtype"})


def radio_recurrence(df):
    return _radio_binary(df, "Recurrence", "recurrence",
                         {"yes"}, {"no"})


def radio_pleural_invasion(df):
    return _radio_binary(
        df, "Pleural invasion (elastic, visceral, or parietal)",
        "pleural_invasion", {"Yes"}, {"No"},
    )


def radio_os2yr(df):
    ct = pd.to_datetime(df["CT Date"], errors="coerce")
    lka = pd.to_datetime(df["Date of Last Known Alive"], errors="coerce")
    ttd = pd.to_numeric(df["Time to Death (days)"], errors="coerce")
    fu_alive = (lka - ct).dt.days
    fu = ttd.where(df["Survival Status"] == "Dead", fu_alive)
    pos = (df["Survival Status"] == "Dead") & (fu <= 730)
    neg = ((df["Survival Status"] == "Dead") & (fu > 730)) | \
          ((df["Survival Status"] == "Alive") & (fu >= 730))
    keep = pos | neg
    out = df.loc[keep, ["Case ID"]].rename(columns={"Case ID": "PatientID"}).copy()
    out["os2yr"] = pos[keep].astype(int).values
    return out, "os2yr"


LUNG1_LABELSETS = [
    ("lung1", "os2yr",          lung1_os2yr),
    ("lung1", "histology",      lung1_histology),
    ("lung1", "stage_advanced", lung1_stage_advanced),
]
RADIO_LABELSETS = [
    ("radio", "egfr",             radio_egfr),
    ("radio", "kras",             radio_kras),
    ("radio", "recurrence",       radio_recurrence),
    ("radio", "pleural_invasion", radio_pleural_invasion),
    ("radio", "os2yr",            radio_os2yr),
]


def build_one(df, dataset, labelset, builder, out_dir, test_frac=TEST_FRAC):
    sub, strat_col = builder(df)
    sub = sub.reset_index(drop=True)
    te_idx = stratified_split(sub[strat_col].values, test_frac, SEED)
    sub["split"] = "train"
    sub.loc[te_idx, "split"] = "test"
    fp = out_dir / f"{dataset}_{labelset}_split.csv"
    sub.to_csv(fp, index=False)
    n_tr = int((sub.split == "train").sum())
    n_te = int((sub.split == "test").sum())
    n_tr_p = int(sub.loc[sub.split == "train", strat_col].sum())
    n_te_p = int(sub.loc[sub.split == "test", strat_col].sum())
    print(f"[{dataset}/{labelset}] n={len(sub)}  test_frac={test_frac:.2f}  "
          f"train={n_tr} ({n_tr_p} pos)  test={n_te} ({n_te_p} pos)  → {fp.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lung1_csv", default=LUNG1_CSV_DEFAULT)
    ap.add_argument("--radio_csv", default=RADIO_CSV_DEFAULT)
    ap.add_argument("--out_dir", default="/path/to/data/tcia/manifests")
    ap.add_argument("--test_frac", type=float, default=TEST_FRAC,
                    help=f"Test-set fraction (default {TEST_FRAC}). Overrides "
                         "the module-level constant for this run.")
    ap.add_argument("--only_datasets", default=None,
                    help="Comma-separated subset of {lung1, radio} to regenerate; "
                         "if unset, both are processed.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    only = set(args.only_datasets.split(",")) if args.only_datasets else None

    if only is None or "lung1" in only:
        lung1 = pd.read_csv(args.lung1_csv)
        for ds, name, fn in LUNG1_LABELSETS:
            build_one(lung1, ds, name, fn, out_dir, test_frac=args.test_frac)

    if only is None or "radio" in only:
        radio = pd.read_csv(args.radio_csv)
        for ds, name, fn in RADIO_LABELSETS:
            build_one(radio, ds, name, fn, out_dir, test_frac=args.test_frac)


if __name__ == "__main__":
    main()

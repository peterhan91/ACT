#!/usr/bin/env python3
"""
Mine PMBB chest / abdomen finding labels from the full radiology reports.

For each volume in an exp2 PMBB manifest we join its full report text, restrict
to the FINDINGS+IMPRESSION sections, and apply the phrase rules (CHEST_18 for
chest, ABD_30 for abdomen). Output is a manifest-aligned label CSV consumed by
exp1 (common.load_dataset / ours.run_dataset).

The chest engine is validated against CT-RATE RadBERT (calibrate_chest.py:
macro-F1 ~0.88). Uncertain ('-1') mentions are mapped to positive by default
(a hedged finding is "on the differential"); a raw 3-way CSV is also written for
inspection.

Usage:
    python mine.py --pool chest      # -> labels/pmbb_chest_test_labels.csv
    python mine.py --pool abd
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import finding_rules as FR
import report_mining as RM

HERE = Path(__file__).resolve().parent
MANIFESTS = HERE.parent.parent / "exp2_retrieval" / "pmbb_manifests"
REPORTS_CSV = Path("/path/to/data/pmbb_ct_volume_reports.csv")
OUT = HERE / "labels"

DATASET = {"chest": "pmbb_chest_test", "abd": "pmbb_abd_test",
           # scaled exp1 pools (supersede the 2,489 pilot)
           "chest_nc": "pmbb_chest_nc", "abd_ce": "pmbb_abd_ce"}
# which finding-rule set (chest=CHEST_18 / abd=ABD_30) each pool uses
REGION = {"chest": "chest", "abd": "abd", "chest_nc": "chest", "abd_ce": "abd"}


def mine_pool(pool: str, uncertain_to: int = 1) -> pd.DataFrame:
    dataset = DATASET[pool]
    rules, exclude = FR.get_rules(REGION[pool])
    findings = list(rules)
    compiled = RM.compile_rules(rules)
    excludes = RM.compile_rules(exclude)

    man = pd.read_csv(MANIFESTS / f"{dataset}.csv")
    rep = pd.read_csv(REPORTS_CSV)[["VolumeName", "impressions"]]  # full report text
    df = man[["VolumeName"]].merge(rep, on="VolumeName", how="left")
    miss = df["impressions"].isna().sum()
    if miss:
        print(f"[warn] {miss}/{len(df)} volumes have no report text (labeled all-absent)")
    df["impressions"] = df["impressions"].fillna("")

    raw = np.zeros((len(df), len(findings)), dtype=np.int64)   # 3-way: 1/0/-1
    for i, txt in enumerate(df["impressions"]):
        st = RM.label_report(txt, compiled, mine_sections=True, excludes=excludes)
        raw[i] = [st[f] for f in findings]

    binary = np.where(raw == RM.UNCERTAIN, uncertain_to,
                      np.where(raw == RM.PRESENT, 1, 0)).astype(np.int64)
    out = pd.DataFrame({"VolumeName": df["VolumeName"].values})
    for j, f in enumerate(findings):
        out[f] = binary[:, j]

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f"{dataset}_labels.csv", index=False)
    raw_df = pd.DataFrame({"VolumeName": df["VolumeName"].values})
    for j, f in enumerate(findings):
        raw_df[f] = raw[:, j]
    raw_df.to_csv(OUT / f"{dataset}_labels_raw3way.csv", index=False)

    # Prevalence + uncertainty summary for spot-checking.
    prev = binary.mean(0)
    n_unc = (raw == RM.UNCERTAIN).sum(0)
    print(f"\n=== {dataset}: mined {len(findings)} findings over {len(df)} volumes ===")
    print(f"{'finding':<34}{'n_pos':>7}{'prev':>8}{'n_uncertain':>13}")
    order = np.argsort(-prev)
    for j in order:
        print(f"{findings[j]:<34}{int(binary[:,j].sum()):>7}{prev[j]:>8.3f}{int(n_unc[j]):>13}")
    print(f"\nwrote {OUT / (dataset + '_labels.csv')}  (+ _labels_raw3way.csv)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["chest", "abd", "chest_nc", "abd_ce", "both"],
                    default="both")
    ap.add_argument("--uncertain", choices=["pos", "neg"], default="pos")
    args = ap.parse_args()
    u = 1 if args.uncertain == "pos" else 0
    pools = ["chest", "abd"] if args.pool == "both" else [args.pool]
    for p in pools:
        mine_pool(p, uncertain_to=u)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Calibrate the CHEST_18 phrase rules against the CT-RATE RadBERT silver labels.

The official CT-RATE labels are produced by a fine-tuned RadBERT classifier. We
have those labels (test_predicted_labels.csv) AND the matching report text
(test_reports.csv) locally, so we can measure how well our phrase-mining
reproduces the canonical labeler per finding, and tune the rules — BEFORE
applying them to PMBB chest (which has no labels).

Usage:
    python calibrate_chest.py [--uncertain {pos,neg,both}]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import finding_rules as FR
import report_mining as RM

CTRATE = Path("/path/to/ACT/model/data/ct_rate")
REPORTS = CTRATE / "test_reports.csv"
LABELS = CTRATE / "test_predicted_labels.csv"


def mine_matrix(reports: pd.Series, compiled, excludes, uncertain_to: int) -> np.ndarray:
    """(N, 18) binary matrix from mining each report's findings+impression text.
    `uncertain_to` maps the -1 (hedged) status to 0 or 1."""
    findings = list(compiled)
    out = np.zeros((len(reports), len(findings)), dtype=np.int64)
    for i, txt in enumerate(reports):
        # CT-RATE fields are already clean findings/impression text -> no section
        # extraction (mine_sections=False).
        st = RM.label_report(txt, compiled, mine_sections=False, excludes=excludes)
        for j, f in enumerate(findings):
            v = st[f]
            out[i, j] = (uncertain_to if v == RM.UNCERTAIN else (1 if v == RM.PRESENT else 0))
    return out


def prf(pred: np.ndarray, gold: np.ndarray):
    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    tn = int(((pred == 0) & (gold == 0)).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if (p == p and r == r and p + r) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, prec=p, rec=r, f1=f1,
                gold_pos=tp + fn, pred_pos=tp + fp)


def evaluate(uncertain_to: int, verbose=True) -> float:
    rep = pd.read_csv(REPORTS)
    lab = pd.read_csv(LABELS)
    df = rep.merge(lab, on="VolumeName", how="inner")
    text = (df["Findings_EN"].fillna("").astype(str) + "\n"
            + df["Impressions_EN"].fillna("").astype(str))
    compiled = RM.compile_rules(FR.CHEST_18)
    excludes = RM.compile_rules(FR.CHEST_EXCLUDE)
    findings = list(compiled)
    pred = mine_matrix(text, compiled, excludes, uncertain_to)
    gold = df[findings].values.astype(np.int64)
    gold = (gold > 0).astype(np.int64)            # ensure binary

    rows, f1s = [], []
    for j, f in enumerate(findings):
        m = prf(pred[:, j], gold[:, j])
        rows.append((f, m))
        if m["f1"] == m["f1"]:
            f1s.append(m["f1"])
    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    micro = prf(pred.ravel(), gold.ravel())
    if verbose:
        print(f"\n=== CHEST_18 phrase-mining vs RadBERT  (uncertain->{uncertain_to}, "
              f"N={len(df)}) ===")
        print(f"{'finding':<36}{'goldP':>6}{'predP':>6}{'prec':>7}{'rec':>7}{'f1':>7}")
        for f, m in rows:
            print(f"{f:<36}{m['gold_pos']:>6}{m['pred_pos']:>6}"
                  f"{m['prec']:>7.3f}{m['rec']:>7.3f}{m['f1']:>7.3f}")
        print(f"{'-'*69}")
        print(f"{'MACRO-F1':<36}{'':>12}{'':>7}{'':>7}{macro_f1:>7.3f}")
        print(f"{'MICRO  (prec/rec/f1)':<36}{'':>12}"
              f"{micro['prec']:>7.3f}{micro['rec']:>7.3f}{micro['f1']:>7.3f}")
    return macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncertain", choices=["pos", "neg", "both"], default="both")
    args = ap.parse_args()
    if args.uncertain in ("neg", "both"):
        f0 = evaluate(0)
    if args.uncertain in ("pos", "both"):
        f1 = evaluate(1)
    if args.uncertain == "both":
        print(f"\n>>> macro-F1: uncertain->neg={f0:.3f}  uncertain->pos={f1:.3f}  "
              f"(pick the higher for mining)")


if __name__ == "__main__":
    main()

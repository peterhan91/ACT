#!/usr/bin/env python3
"""Aggregate exp1 zero-shot AUROC on the PMBB pools across all models.
Reads every <model>/results/summary.csv, keeps pmbb_* rows, writes a table."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ORDER = ["ours", "colipri", "ctclip", "merlin", "m3dclip",
         "medsiglip", "biomedclip", "openai_clip"]
POOLS = ["pmbb_chest_test", "pmbb_abd_test"]


def main():
    rows = []
    for csv in sorted(HERE.glob("*/results/summary.csv")):
        df = pd.read_csv(csv)
        df["model"] = df.get("model", csv.parent.parent.name)
        df["model"] = df["model"].fillna(csv.parent.parent.name)
        rows.append(df)
    if not rows:
        raise SystemExit("no summaries yet")
    df = pd.concat(rows, ignore_index=True)
    df = df[df["dataset"].isin(POOLS)]
    if df.empty:
        raise SystemExit("no pmbb rows yet")
    df.to_csv(HERE / "results_pmbb_exp1.csv", index=False)
    # best mode per (model,dataset)
    best = (df.sort_values("mean_auc")
              .groupby(["model", "dataset"], as_index=False).last())
    t = best.pivot_table(index="model", columns="dataset", values="mean_auc", aggfunc="last")
    order = [m for m in ORDER if m in t.index] + [m for m in t.index if m not in ORDER]
    t = t.reindex(index=order, columns=[p for p in POOLS if p in t.columns])
    L = ["# Exp 1 — PMBB zero-shot finding classification (mean AUROC)", "",
         "Labels phrase-mined from reports (see pmbb_labels/). chest=18 CT-RATE "
         "findings, abd=30 Merlin findings. `ours` = h5 (our pipeline); 3D baselines "
         "= native NIfTI; 2D = h5 + own per-slice pipeline. Best per column **bold**.", "",
         "| model | chest (18) | abd (30) |", "|---|---|---|"]
    bestv = {p: t[p].max(skipna=True) for p in t.columns}
    for m in t.index:
        cells = []
        for p in t.columns:
            v = t.loc[m, p]
            cells.append("—" if pd.isna(v) else (f"**{v:.4f}**" if v == bestv[p] else f"{v:.4f}"))
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    (HERE / "RESULTS_PMBB_EXP1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {HERE/'RESULTS_PMBB_EXP1.md'} + results_pmbb_exp1.csv")


if __name__ == "__main__":
    main()

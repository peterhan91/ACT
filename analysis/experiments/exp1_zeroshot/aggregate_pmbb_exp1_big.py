#!/usr/bin/env python3
"""Aggregate the SCALED PMBB exp1 zero-shot results (pmbb_chest_nc / pmbb_abd_ce)
across all models. Writes:
  * RESULTS_PMBB_EXP1_BIG.md      — mean-AUROC table (best mode per model/pool)
  * results_pmbb_exp1_big.csv     — every row (model,dataset,mode,n,mean_auc,sec)
  * per_class_pmbb_exp1_big.csv   — tidy model x finding x pool per-label AUROC
COLIPRI excluded by design.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ORDER = ["ours", "ctclip", "merlin", "m3dclip", "medsiglip", "biomedclip", "openai_clip"]
POOLS = ["pmbb_chest_nc", "pmbb_abd_ce"]


def main():
    rows, per_class = [], []
    # Source mean_auc + per-class straight from the authoritative per-run results.json
    # (NOT summary.csv — a run that crashed on a later dataset never wrote its
    # summary.csv, e.g. the native models finished chest_nc then died on abd_ce, so
    # their chest_nc rows are only in results.json).
    for jf in sorted(HERE.glob("*/results/*results.json")):
        d = json.load(open(jf))
        if d.get("dataset") not in POOLS:
            continue
        model = d.get("model") or jf.parts[-3]
        if d.get("mean_auc") is not None:
            rows.append({"model": model, "dataset": d["dataset"], "mode": d.get("mode"),
                         "n_volumes": d.get("n_volumes"), "n_labels": d.get("n_labels"),
                         "mean_auc": d["mean_auc"]})
        for lab, auc in (d.get("per_label_auc") or {}).items():
            per_class.append({"model": model, "dataset": d["dataset"],
                              "mode": d.get("mode"), "finding": lab, "auc": auc})
    df = pd.DataFrame(rows)
    df = df[df["dataset"].isin(POOLS)]
    if df.empty:
        raise SystemExit("no scaled-pool rows yet")
    df.to_csv(HERE / "results_pmbb_exp1_big.csv", index=False)
    if per_class:
        pd.DataFrame(per_class).to_csv(HERE / "per_class_pmbb_exp1_big.csv", index=False)

    best = (df.sort_values("mean_auc")
              .groupby(["model", "dataset"], as_index=False).last())
    t = best.pivot_table(index="model", columns="dataset", values="mean_auc", aggfunc="last")
    order = [m for m in ORDER if m in t.index] + [m for m in t.index if m not in ORDER]
    t = t.reindex(index=order, columns=[p for p in POOLS if p in t.columns])
    nvol = best.set_index(["model", "dataset"])["n_volumes"].to_dict()
    L = ["# Exp 1 — SCALED PMBB zero-shot finding classification (mean AUROC)", "",
         "Supersedes the 2,489 pilot. Pools: **pmbb_chest_nc** (non-contrast chest, "
         "9,097) + **pmbb_abd_ce** (contrast/Merlin-style abd, 14,290), 1 vol/patient, "
         "labels phrase-mined from reports. `ours`=h5 pipeline; 3D baselines=native "
         "NIfTI; 2D=h5+own per-slice. COLIPRI dropped. Best per column **bold**.", "",
         "| model | chest_nc (18) | abd_ce (30) |", "|---|---|---|"]
    bestv = {p: t[p].max(skipna=True) for p in t.columns}
    for m in t.index:
        cells = []
        for p in t.columns:
            v = t.loc[m, p]
            cells.append("—" if pd.isna(v) else (f"**{v:.4f}**" if v == bestv[p] else f"{v:.4f}"))
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    (HERE / "RESULTS_PMBB_EXP1_BIG.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote RESULTS_PMBB_EXP1_BIG.md + results_pmbb_exp1_big.csv + "
          f"per_class_pmbb_exp1_big.csv")


if __name__ == "__main__":
    main()

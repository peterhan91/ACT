#!/usr/bin/env python3
"""Aggregate the INSPECT-phenotype cross-model zero-shot table.

Collects every baseline's inspect_pheno results.json (per_label_auc keyed by the
221 phecode_str names) + ours (plain/openai/sfr/harrier from outputs/v1/external),
and reports mean test AUROC over (a) all 221 phecodes and (b) the 97 level-1
(integer) phecodes. Writes inspect_pheno_summary.{csv,md} + STATUS.md.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                       # inspect_pheno/
EXP1 = HERE.parent                                            # exp1_zeroshot/
OURS = Path("/path/to/ACT/analysis/outputs/v1/external")

ph = pd.read_csv(HERE / "phecodes.csv", dtype=str)
ph["phecode_str"] = ph["phecode_str"].str.strip()
is_int = {s: ("." not in c) for c, s in zip(ph["phecode"], ph["phecode_str"])}
N_L1 = sum(is_int.values())


def summarize(per_label: dict):
    """(mean over all scored 221, mean over level-1 integer subset, n_scored)."""
    vals = [(s, a) for s, a in per_label.items() if a == a]            # drop NaN
    all_m = float(np.mean([a for _, a in vals])) if vals else float("nan")
    l1 = [a for s, a in vals if is_int.get(s, False)]
    l1_m = float(np.mean(l1)) if l1 else float("nan")
    return all_m, l1_m, len(vals), len(l1)


rows = []

# --- ours (canon v1) — multiple modes ---
for mode in ["plain", "openai", "sfr", "harrier", "f2llm", "gteqwen2"]:
    p = OURS / f"phenotype__test__{mode}__results.json"
    if p.exists():
        d = json.load(open(p))
        a, l1, n, nl1 = summarize(d["per_label_auc"])
        tag = "ours (canon v1) " + ("[NO proj]" if mode == "plain" else f"[{mode} proj]")
        rows.append(dict(model=tag, mode=mode, n_vol=d["n_volumes"],
                         auc_221=a, auc_l1_97=l1, n_scored=n, n_l1=nl1))

# --- baselines ---
for mdl in ["merlin", "ctclip", "m3dclip", "medsiglip", "biomedclip", "openai_clip"]:
    for p in sorted((EXP1 / mdl / "results").glob("inspect_pheno__*__results.json")):
        d = json.load(open(p))
        a, l1, n, nl1 = summarize(d["per_label_auc"])
        rows.append(dict(model=mdl, mode=d.get("mode", "native"), n_vol=d["n_volumes"],
                         auc_221=a, auc_l1_97=l1, n_scored=n, n_l1=nl1))

df = pd.DataFrame(rows).sort_values("auc_221", ascending=False).reset_index(drop=True)
df.to_csv(HERE / "inspect_pheno_summary.csv", index=False)

md = ["# INSPECT CTPA + EHR phenotypes — zero-shot cross-model (test AUROC)\n",
      f"2,612 test volumes · 221 phecodes (level-1 integer subset = {N_L1}). "
      "Baselines = native original pipeline; ours = CLEAR projection modes.\n",
      "| model | mode | n_vol | mean AUROC (221) | mean AUROC (level-1 97) | n_scored |",
      "|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    md.append(f"| {r['model']} | {r['mode']} | {int(r['n_vol'])} | "
              f"{r['auc_221']:.4f} | {r['auc_l1_97']:.4f} | {int(r['n_scored'])} |")
(HERE / "inspect_pheno_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nwrote -> {HERE/'inspect_pheno_summary.csv'} and .md")

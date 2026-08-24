#!/usr/bin/env python3
"""Aggregate the 20-seed v1/linear_<LLM> results into mean ± std summaries.

Reads per-seed result JSONs written by exp_linear_ct.py / exp_phenotype_ct.py
under --out_suffix __seed{N}__lr{BEST_LR}, computes mean and std of the
overall mean AUC and per-label AUCs across the 20 seeds, writes:

    outputs/v1/cbm/linear_<LLM>__20seeds_summary.csv         (CT-RATE 18)
    outputs/v1/external/pe_pheno__linear_<LLM>__20seeds_summary.csv
    outputs/v1/external/phenotype__linear_<LLM>__20seeds_summary.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def collect_jsons(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def summarize(per_seed: list[dict], mean_key: str, per_label_key: str) -> tuple[dict, pd.DataFrame]:
    """Returns (overall_summary_dict, per_label_dataframe)."""
    overall = np.array([float(d[mean_key]) for d in per_seed])
    per_label = defaultdict(list)
    for d in per_seed:
        for lbl, v in d[per_label_key].items():
            per_label[lbl].append(np.nan if v is None else float(v))
    rows = []
    for lbl, vs in per_label.items():
        arr = np.array(vs, dtype=float)
        rows.append({
            "label": lbl,
            "n_seeds": int(np.isfinite(arr).sum()),
            "mean_auc": float(np.nanmean(arr)),
            "std_auc": float(np.nanstd(arr, ddof=1)) if np.isfinite(arr).sum() > 1 else 0.0,
            "min_auc": float(np.nanmin(arr)),
            "max_auc": float(np.nanmax(arr)),
        })
    df_per_label = pd.DataFrame(rows).sort_values("mean_auc", ascending=False)
    overall_dict = {
        "n_seeds": int(len(per_seed)),
        "overall_mean_auc": float(overall.mean()),
        "overall_std_auc": float(overall.std(ddof=1)) if len(overall) > 1 else 0.0,
        "overall_min_auc": float(overall.min()),
        "overall_max_auc": float(overall.max()),
    }
    return overall_dict, df_per_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="openai",
                    help="LLM tag — must match the linear_<llm> filenames")
    ap.add_argument("--ctrate18_bestlr", default=None)
    ap.add_argument("--pe_bestlr", default=None)
    ap.add_argument("--pheno_bestlr", default=None)
    ap.add_argument("--cbm_pheno_bestlr", default=None,
                    help="If set, also aggregate phenotype CBM seeds at this LR.")
    ap.add_argument("--rsna2023_bestlr", default=None,
                    help="If set, aggregate RSNA-2023 trauma linear_<llm> seeds at this LR.")
    args = ap.parse_args()
    LLM = args.llm

    # CT-RATE-18 — exp_linear_ct.py result format: test_mean_auc + test_per_label_auc
    if args.ctrate18_bestlr:
        pat = f"outputs/v1/cbm/linear_{LLM}__seed*__lr{args.ctrate18_bestlr}__test_results.json"
        per_seed = collect_jsons(pat)
    else:
        print("[ctrate18] skipped (no --ctrate18_bestlr)")
        per_seed = []
        pat = None
    if not per_seed and pat is not None:
        print(f"[ctrate18] no files match {pat}")
    elif per_seed:
        overall, df = summarize(per_seed, "test_mean_auc", "test_per_label_auc")
        out_csv = f"outputs/v1/cbm/linear_{LLM}__20seeds_summary.csv"
        df.to_csv(out_csv, index=False)
        out_json = f"outputs/v1/cbm/linear_{LLM}__20seeds_overall.json"
        with open(out_json, "w") as f:
            json.dump({"llm": LLM, "lr": args.ctrate18_bestlr, **overall}, f, indent=2)
        print(f"[ctrate18/{LLM}] {len(per_seed)} seeds  →  test_mean_auc = "
              f"{overall['overall_mean_auc']:.4f} ± {overall['overall_std_auc']:.4f}")
        print(f"  → {out_csv}\n  → {out_json}")

    # PE-3 — exp_phenotype_ct.py format: mean_auc + per_label_auc
    if args.pe_bestlr:
        pat = f"outputs/v1/external/pe_pheno__test__linear_{LLM}__seed*__lr{args.pe_bestlr}__results.json"
        per_seed = collect_jsons(pat)
    else:
        print("[pe] skipped (no --pe_bestlr)")
        per_seed = []
        pat = None
    if not per_seed and pat is not None:
        print(f"[pe] no files match {pat}")
    elif per_seed:
        overall, df = summarize(per_seed, "mean_auc", "per_label_auc")
        out_csv = f"outputs/v1/external/pe_pheno__linear_{LLM}__20seeds_summary.csv"
        df.to_csv(out_csv, index=False)
        out_json = f"outputs/v1/external/pe_pheno__linear_{LLM}__20seeds_overall.json"
        with open(out_json, "w") as f:
            json.dump({"llm": LLM, "lr": args.pe_bestlr, **overall}, f, indent=2)
        print(f"[pe/{LLM}]       {len(per_seed)} seeds  →  test_mean_auc = "
              f"{overall['overall_mean_auc']:.4f} ± {overall['overall_std_auc']:.4f}")
        print(f"  → {out_csv}\n  → {out_json}")

    # phenotype-221 — same format as PE
    if args.pheno_bestlr:
        pat = f"outputs/v1/external/phenotype__test__linear_{LLM}__seed*__lr{args.pheno_bestlr}__results.json"
        per_seed = collect_jsons(pat)
    else:
        print("[phenotype] skipped (no --pheno_bestlr)")
        per_seed = []
        pat = None
    if not per_seed and pat is not None:
        print(f"[phenotype] no files match {pat}")
    elif per_seed:
        overall, df = summarize(per_seed, "mean_auc", "per_label_auc")
        out_csv = f"outputs/v1/external/phenotype__linear_{LLM}__20seeds_summary.csv"
        df.to_csv(out_csv, index=False)
        out_json = f"outputs/v1/external/phenotype__linear_{LLM}__20seeds_overall.json"
        with open(out_json, "w") as f:
            json.dump({"llm": LLM, "lr": args.pheno_bestlr, **overall}, f, indent=2)
        print(f"[phenotype/{LLM}] {len(per_seed)} seeds  →  test_mean_auc = "
              f"{overall['overall_mean_auc']:.4f} ± {overall['overall_std_auc']:.4f}")
        print(f"  → {out_csv}\n  → {out_json}")

    # RSNA-2023 trauma — exp_rsna2023_ct.py format: mean_auc + per_label_auc.
    if args.rsna2023_bestlr:
        pat = f"outputs/v1/external/rsna2023__test__linear_{LLM}__seed*__lr{args.rsna2023_bestlr}__results.json"
        per_seed = collect_jsons(pat)
    else:
        print("[rsna2023] skipped (no --rsna2023_bestlr)")
        per_seed = []
        pat = None
    if not per_seed and pat is not None:
        print(f"[rsna2023] no files match {pat}")
    elif per_seed:
        overall, df = summarize(per_seed, "mean_auc", "per_label_auc")
        out_csv = f"outputs/v1/external/rsna2023__linear_{LLM}__20seeds_summary.csv"
        df.to_csv(out_csv, index=False)
        out_json = f"outputs/v1/external/rsna2023__linear_{LLM}__20seeds_overall.json"
        with open(out_json, "w") as f:
            json.dump({"llm": LLM, "lr": args.rsna2023_bestlr, **overall}, f, indent=2)
        print(f"[rsna2023/{LLM}]  {len(per_seed)} seeds  →  test_mean_auc = "
              f"{overall['overall_mean_auc']:.4f} ± {overall['overall_std_auc']:.4f}")
        print(f"  → {out_csv}\n  → {out_json}")

    # phenotype-221 / CBM head — independent of --llm; gated on --cbm_pheno_bestlr.
    if args.cbm_pheno_bestlr:
        pat = f"outputs/v1/external/phenotype__test__cbm__seed*__lr{args.cbm_pheno_bestlr}__results.json"
        per_seed = collect_jsons(pat)
        if not per_seed:
            print(f"[phenotype/cbm] no files match {pat}")
        else:
            overall, df = summarize(per_seed, "mean_auc", "per_label_auc")
            out_csv = f"outputs/v1/external/phenotype__cbm__20seeds_summary.csv"
            df.to_csv(out_csv, index=False)
            out_json = f"outputs/v1/external/phenotype__cbm__20seeds_overall.json"
            with open(out_json, "w") as f:
                json.dump({"method": "cbm", "lr": args.cbm_pheno_bestlr, **overall}, f, indent=2)
            print(f"[phenotype/cbm]    {len(per_seed)} seeds  →  test_mean_auc = "
                  f"{overall['overall_mean_auc']:.4f} ± {overall['overall_std_auc']:.4f}")
            print(f"  → {out_csv}\n  → {out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 4 — External validation on INSPECT (train + valid + test).

Two label families are evaluated for each split:

    A) The 18 RadBERT-predicted CT-RATE pathology labels  (`predicted_labels`)
       — same schema as the in-domain CT-RATE evaluation; checks transferability.
    B) The 3 native INSPECT PE-task labels                 (`pe_labels`)
       — pe_positive / pe_acute / pe_subsegmentalonly. These are the labels
       the INSPECT paper actually defines as benchmark tasks.

The user originally asked to use labels from
`https://github.com/peterhan91/phenotype_labels`, which currently 404s. If/when
that repo is shared, drop its labels CSV next to the existing pe_labels CSVs in
`/path/to/ACT/model/data/inspect/` and rerun this script with
`--phenotype_labels_csv <path>` and `--phenotype_label_names <comma-list>`.

Outputs (per split & method, in outputs/external/):
    inspect_<split>__<labelset>__<method>__results.json
    inspect_<split>__<labelset>__<method>__probs.npy
    summary.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clear3d.data import load_pe_split, load_split
from clear3d.features import encode_images, encode_texts_clip, free_3d_clip, load_3d_clip
from clear3d.llm_prompts import encode_label_texts
from clear3d.metrics import mean_auc, per_label_auc, softmax_pos_neg
from clear3d.paths import (
    EXTERNAL_DIR, LABELS_18, LABELS_PE, cache_label_emb,
)
from clear3d.projection import get_or_compute_llm_repr


# ----------------------------------------------------------------------------
# Custom-label support: if the user has a different label list than the 18 or
# the 3 PE ones, supply --custom_label_names and we'll embed those prompts on
# the fly (cached per signature in outputs/cache/label_emb.<llm>.<sig>.npz).
# ----------------------------------------------------------------------------
def _label_emb_path(llm: str, sig: str) -> Path:
    return cache_label_emb(llm).with_name(f"label_emb.{llm}.{sig}.npz")


def get_label_embeddings_for(labels: list[str], llm: str) -> dict:
    """Like `clear3d.llm_prompts.get_label_embeddings`, but for an arbitrary
    label list. Cache key incorporates a stable hash of the label list."""
    import hashlib
    sig = hashlib.sha1("|".join(labels).encode("utf-8")).hexdigest()[:10]
    out = _label_emb_path(llm, sig)
    if out.exists():
        z = np.load(out, allow_pickle=True)
        return {"pos": z["pos"], "neg": z["neg"], "labels": list(z["labels"])}

    pos_texts = [lbl.lower() for lbl in labels]
    neg_texts = [f"no {lbl.lower()}" for lbl in labels]
    print(f"[external:label_emb:{llm}] encoding {2*len(labels)} prompts")
    all_emb = encode_label_texts(llm, pos_texts + neg_texts)
    pos = all_emb[: len(labels)]
    neg = all_emb[len(labels):]
    np.savez(out, pos=pos, neg=neg, labels=np.asarray(labels, dtype=object))
    return {"pos": pos, "neg": neg, "labels": labels}


def run_for(dataset: str, labelset: str, methods: list[str],
            args, summary_rows: list[dict]) -> None:
    print(f"\n========== {dataset} | labelset={labelset} ==========")
    if labelset == "predicted":
        df = load_split(dataset)
        labels = LABELS_18
    elif labelset == "pe":
        df = load_pe_split(dataset)
        labels = LABELS_PE
    else:
        raise ValueError(f"unknown labelset {labelset}")
    y_true = df[labels].values.astype(np.int64)
    print(f"  {len(df)} volumes; {len(labels)} labels")

    img_feats = encode_images(dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers)

    # 1) plain 3D-CLIP (uses the 3D-CLIP text tower)
    if "plain" in methods:
        t0 = time.time()
        pos_clip = encode_texts_clip([lbl.lower() for lbl in labels])
        neg_clip = encode_texts_clip([f"no {lbl.lower()}" for lbl in labels])
        probs = softmax_pos_neg(img_feats, pos_clip, neg_clip)
        auc = per_label_auc(probs, y_true, labels)
        m = mean_auc(auc)
        save(dataset, labelset, "plain", probs, auc, m, len(df), labels, time.time()-t0,
             summary_rows)

    # 2) CLEAR projection w/ SFR / Harrier / OpenAI
    free_3d_clip()
    from clear3d.llm_prompts import OpenAIUnavailable
    for llm in ("sfr", "harrier", "openai", "f2llm", "gteqwen2"):
        if llm not in methods:
            continue
        t0 = time.time()
        llm_repr = get_or_compute_llm_repr(dataset, llm, img_feats)
        try:
            if labelset == "predicted":
                from clear3d.llm_prompts import get_label_embeddings
                label_emb = get_label_embeddings(llm)
            else:
                label_emb = get_label_embeddings_for(labels, llm)
        except OpenAIUnavailable as e:
            print(f"[skip {llm}] {e}")
            continue
        probs = softmax_pos_neg(llm_repr, label_emb["pos"], label_emb["neg"])
        auc = per_label_auc(probs, y_true, labels)
        m = mean_auc(auc)
        save(dataset, labelset, llm, probs, auc, m, len(df), labels, time.time()-t0,
             summary_rows)


def save(dataset, labelset, method, probs, per_label, m, n, labels, secs, summary_rows):
    out_json = EXTERNAL_DIR / f"{dataset}__{labelset}__{method}__results.json"
    out_probs = EXTERNAL_DIR / f"{dataset}__{labelset}__{method}__probs.npy"
    np.save(out_probs, probs)
    payload = {
        "dataset": dataset, "labelset": labelset, "method": method,
        "n_volumes": n, "n_labels": len(labels),
        "labels": labels, "mean_auc": m, "per_label_auc": per_label,
        "seconds": secs,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [{method}] mean AUC = {m:.4f}  ({secs:.0f}s)  → {out_json.name}")
    summary_rows.append(payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["inspect_train", "inspect_valid", "inspect_test"])
    ap.add_argument("--labelsets", nargs="*",
                    default=["predicted", "pe"])
    ap.add_argument("--methods", nargs="*",
                    default=["plain", "sfr", "harrier", "openai"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    summary_rows: list[dict] = []
    for ds in args.datasets:
        for ls in args.labelsets:
            run_for(ds, ls, args.methods, args, summary_rows)

    # consolidated summary
    rows = []
    for r in summary_rows:
        row = {
            "dataset": r["dataset"], "labelset": r["labelset"],
            "method": r["method"], "n_volumes": r["n_volumes"],
            "mean_auc": r["mean_auc"],
        }
        row.update(r["per_label_auc"])
        rows.append(row)
    df = pd.DataFrame(rows)
    out = EXTERNAL_DIR / "summary.csv"
    if out.exists():
        old = pd.read_csv(out)
        df = (pd.concat([old, df], ignore_index=True)
                .drop_duplicates(subset=["dataset", "labelset", "method"], keep="last"))
    df.to_csv(out, index=False)
    print(f"\n=== Summary → {out} ===")
    print(df[["dataset", "labelset", "method", "mean_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()

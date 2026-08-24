#!/usr/bin/env python3
"""
Step 1 — CLEAR-3D zero-shot evaluation.

For each (dataset, method) pair:
    method = plain     : 3D-CLIP-only, replicates eval_all.py's softmax(pos/neg).
    method = sfr       : project image features into SFR-Mistral concept space,
                         softmax against SFR-Mistral pos/neg label embeddings.
    method = harrier   : same with Harrier-OSS-27B.

Outputs (per dataset & method):
    outputs/zeroshot/<dataset>__<method>__results.json
    outputs/zeroshot/<dataset>__<method>__probs.npy

A single summary CSV is also written:
    outputs/zeroshot/summary.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clear3d.data import load_split
from clear3d.features import encode_images, encode_texts_clip, free_3d_clip, load_3d_clip
from clear3d.llm_prompts import get_label_embeddings
from clear3d.metrics import mean_auc, per_label_auc, softmax_pos_neg
from clear3d.paths import LABELS_18, ZEROSHOT_DIR
from clear3d.projection import get_or_compute_llm_repr, load_concept_bank


def run_plain(dataset: str, img_feats: np.ndarray, y_true: np.ndarray) -> dict:
    """3D-CLIP zero-shot: encode the 18 labels (pos/neg) with the 3D-CLIP text
    tower itself and softmax. Same recipe as `eval_all.py:encode_text_features`.
    """
    pos_texts = [lbl.lower() for lbl in LABELS_18]
    neg_texts = [f"no {lbl.lower()}" for lbl in LABELS_18]
    pos = encode_texts_clip(pos_texts)
    neg = encode_texts_clip(neg_texts)
    probs = softmax_pos_neg(img_feats, pos, neg)
    auc = per_label_auc(probs, y_true, LABELS_18)
    return {"probs": probs, "per_label_auc": auc, "mean_auc": mean_auc(auc)}


def run_clear(dataset: str, llm: str, img_feats: np.ndarray, y_true: np.ndarray):
    """CLEAR projection + LLM softmax. Returns None if the LLM's label
    embeddings can't be obtained (no cache + no API), so the caller can skip."""
    from clear3d.llm_prompts import OpenAIUnavailable
    llm_repr = get_or_compute_llm_repr(dataset, llm, img_feats)
    try:
        label_emb = get_label_embeddings(llm)
    except OpenAIUnavailable as e:
        print(f"[skip {llm}] {e}")
        return None
    probs = softmax_pos_neg(llm_repr, label_emb["pos"], label_emb["neg"])
    auc = per_label_auc(probs, y_true, LABELS_18)
    return {"probs": probs, "per_label_auc": auc, "mean_auc": mean_auc(auc)}


def save_result(dataset: str, method: str, res: dict, n_volumes: int):
    out_json = ZEROSHOT_DIR / f"{dataset}__{method}__results.json"
    out_probs = ZEROSHOT_DIR / f"{dataset}__{method}__probs.npy"
    np.save(out_probs, res["probs"])
    payload = {
        "dataset": dataset,
        "method": method,
        "n_volumes": n_volumes,
        "n_labels": len(LABELS_18),
        "labels": LABELS_18,
        "mean_auc": res["mean_auc"],
        "per_label_auc": res["per_label_auc"],
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {out_json}  mean_auc={res['mean_auc']:.4f}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["ctrate_test"],
                    help="splits to evaluate (e.g. ctrate_test inspect_test)")
    ap.add_argument("--methods", nargs="*",
                    default=["plain", "sfr", "harrier", "openai"],
                    help="zeroshot methods to run")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    summary_rows = []
    for dataset in args.datasets:
        df = load_split(dataset)
        y_true = df[LABELS_18].values.astype(np.int64)
        print(f"\n========== {dataset}: {len(df)} volumes, {len(LABELS_18)} labels ==========")

        # 1) image features (cached) — also needed to drive the bank projection
        img_feats = encode_images(dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  overwrite=args.overwrite)

        # 2) plain 3D-CLIP zero-shot
        if "plain" in args.methods:
            t0 = time.time()
            res = run_plain(dataset, img_feats, y_true)
            payload = save_result(dataset, "plain", res, n_volumes=len(df))
            payload["seconds"] = time.time() - t0
            summary_rows.append(payload)

        # Free 3D-CLIP before loading SFR/Harrier — sentence-transformers will
        # try to grab `device_map=auto` and we need the GPU.
        free_3d_clip()

        # 3) CLEAR projection w/ SFR / Harrier / OpenAI / F2LLM / gte-Qwen2
        for llm in ("sfr", "harrier", "openai", "f2llm", "gteqwen2"):
            if llm not in args.methods:
                continue
            t0 = time.time()
            res = run_clear(dataset, llm, img_feats, y_true)
            if res is None:
                continue
            payload = save_result(dataset, llm, res, n_volumes=len(df))
            payload["seconds"] = time.time() - t0
            summary_rows.append(payload)

    if summary_rows:
        summary_df = pd.DataFrame([
            {
                "dataset": r["dataset"],
                "method": r["method"],
                "n_volumes": r["n_volumes"],
                "mean_auc": r["mean_auc"],
                **r["per_label_auc"],
            }
            for r in summary_rows
        ])
        summary_path = ZEROSHOT_DIR / "summary.csv"
        # Append-friendly write: if file exists, merge & dedupe on (dataset, method).
        if summary_path.exists() and not args.overwrite:
            old = pd.read_csv(summary_path)
            summary_df = (
                pd.concat([old, summary_df], ignore_index=True)
                  .drop_duplicates(subset=["dataset", "method"], keep="last")
            )
        summary_df.to_csv(summary_path, index=False)
        print(f"\n=== Summary → {summary_path} ===")
        print(summary_df[["dataset", "method", "mean_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()

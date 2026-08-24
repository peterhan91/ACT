#!/usr/bin/env python3
"""
CLEAR_CT_EXPS / exp1_zeroshot — OUR model.

Zero-shot classification of CT findings with the canonical v1 3D-CLIP
checkpoint clip_3d_ctrate_merlin_v1 (dinov2 vitb14, transformer fusion depth 2;
RUN_TAG=v1). This driver is a thin orchestrator over the existing
`clip_3d_concepts/clear3d` package — it reuses that package's model loader,
image encoder, projection, label-prompt embeddings, on-disk caches, and AUC
metrics. Nothing is reimplemented here. Backbone is selected by the CLIP3D_BEST
/ CLIP3D_RUN_TAG env vars that run.sh exports.

Two zero-shot modes (no training):
    plain   — encode pos/neg label prompts with the 3D-CLIP text tower itself,
              then softmax(sim+, sim-).                 (eval_all.py recipe)
    openai  — project image features into text-embedding-3-large concept space
              (CLEAR projection), softmax against OpenAI pos/neg label embeddings.

Three datasets (held-out test splits):
    ctrate_test    — 18 CT-RATE pathology findings   (in-domain)
    radchest       — same 18 findings                (external OOD, RAD-ChestCT)
    rsna2023_test  — 9 RSNA-2023 abdominal-trauma findings

Outputs land in this folder's results/ (NOT in the clip_3d_concepts repo):
    <dataset>__<mode>__results.json   per-label + mean AUC + provenance
    <dataset>__<mode>__probs.npy      (N, n_labels) softmax-positive probs
    summary.csv / summary.md          consolidated table across all runs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Import the shared CLEAR-3D package from the clip_3d_concepts repo.
REPO = os.environ.setdefault("CLEAR3D_REPO", "/path/to/ACT/analysis")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from clear3d.data import load_split                                    # noqa: E402
from clear3d.features import (                                         # noqa: E402
    encode_images, encode_texts_clip, free_3d_clip,
)
from clear3d.llm_prompts import get_label_embeddings, OpenAIUnavailable  # noqa: E402
from clear3d.metrics import mean_auc, per_label_auc, softmax_pos_neg   # noqa: E402
from clear3d.projection import get_or_compute_llm_repr                 # noqa: E402
from clear3d.paths import (                                            # noqa: E402
    CLIP3D_BEST, RUN_TAG, LABELS_18,
    LABELS_RSNA2023_TRAUMA, LABELS_RSNA2023_TRAUMA_PROMPTS,
)
# Reuse the repo's arbitrary-label OpenAI embedding helper (cached by label-list
# hash) for the trauma prompts; the 18-label set uses the canonical cache.
from run_external_ct import get_label_embeddings_for                   # noqa: E402

MODEL = "ours"  # row label in the cross-model comparison table
# Default results/ dir; override with OURS_RESULTS_DIR to keep per-checkpoint runs
# from clobbering the canonical v1 outputs.
RESULTS = Path(os.environ.get("OURS_RESULTS_DIR",
                              str(Path(__file__).resolve().parent / "results")))
RESULTS.mkdir(parents=True, exist_ok=True)

# dataset -> (label columns for y_true, prompt strings to embed, is_18label).
# For the 18-label sets the prompt IS the lowercased label (eval_all convention);
# for trauma the prompt is the curated human-readable phrase.
SPECS = {
    "ctrate_test":   (LABELS_18, [l.lower() for l in LABELS_18], True),
    "radchest":      (LABELS_18, [l.lower() for l in LABELS_18], True),
    "rsna2023_test": (LABELS_RSNA2023_TRAUMA,
                      list(LABELS_RSNA2023_TRAUMA_PROMPTS), False),
}
# PMBB classification pools: labels are phrase-mined (see ../pmbb_labels). chest
# reuses the 18 CT-RATE labels (is_18 -> canonical OpenAI cache); abd uses the 30
# Merlin findings (is_18=False -> hash-keyed OpenAI cache for the openai mode).
PMBB_DATASETS = ["pmbb_chest_test", "pmbb_abd_test", "pmbb_chest_nc", "pmbb_abd_ce"]
PMBB_LABELS_DIR = Path(__file__).resolve().parent.parent / "pmbb_labels" / "labels"


def _load_pmbb_labels(name: str, df):
    lab = pd.read_csv(PMBB_LABELS_DIR / f"{name}_labels.csv")
    labels = [c for c in lab.columns if c != "VolumeName"]
    merged = df[["VolumeName"]].merge(lab, on="VolumeName", how="left")
    if merged[labels].isna().any().any():
        raise ValueError(f"{name}: mined-label join incomplete")
    y = merged[labels].values.astype(np.int64)
    return y, labels, [l.lower() for l in labels]


def _openai_label_emb(prompts: list[str], is_18: bool):
    """(pos, neg) OpenAI label embeddings, from cache. 18-label set hits the
    canonical label_emb.openai.npz; trauma hits the hash-keyed cache."""
    if is_18:
        e = get_label_embeddings("openai")
    else:
        e = get_label_embeddings_for(prompts, "openai")
    return e["pos"], e["neg"]


def _save(ds: str, mode: str, probs: np.ndarray, y: np.ndarray,
          labels: list[str], n: int) -> dict:
    auc = per_label_auc(probs, y, labels)
    m = mean_auc(auc)
    np.save(RESULTS / f"{ds}__{mode}__probs.npy", probs)
    payload = {
        "model": MODEL,
        "config": CLIP3D_BEST,
        "run_tag": RUN_TAG,
        "dataset": ds,
        "mode": mode,
        "n_volumes": int(n),
        "n_labels": len(labels),
        "labels": labels,
        "mean_auc": m,
        "per_label_auc": auc,
    }
    (RESULTS / f"{ds}__{mode}__results.json").write_text(json.dumps(payload, indent=2))
    print(f"[{MODEL}/{ds}/{mode}] mean AUC = {m:.4f}  (n={n}, {len(labels)} labels)")
    return {"model": MODEL, "dataset": ds, "mode": mode,
            "n_volumes": int(n), "n_labels": len(labels), "mean_auc": m}


def run_dataset(ds: str, modes: list[str], batch_size: int, num_workers: int,
                summary: list[dict]) -> None:
    df = load_split(ds)
    if ds.startswith("pmbb_"):
        y, labels, prompts = _load_pmbb_labels(ds, df)
        is_18 = ds in ("pmbb_chest_test", "pmbb_chest_nc")
    else:
        labels, prompts, is_18 = SPECS[ds]
        y = df[labels].values.astype(np.int64)
    print(f"\n========== {ds}: {len(df)} volumes × {len(labels)} labels ==========")

    # Image features (cached per dataset; only rsna2023_test encodes fresh).
    img = encode_images(ds, batch_size=batch_size, num_workers=num_workers)

    # --- plain: 3D-CLIP text tower ---
    if "plain" in modes:
        t0 = time.time()
        pos = encode_texts_clip([p.lower() for p in prompts])
        neg = encode_texts_clip([f"no {p.lower()}" for p in prompts])
        probs = softmax_pos_neg(img, pos, neg)
        row = _save(ds, "plain", probs, y, labels, len(df))
        row["seconds"] = round(time.time() - t0, 1)
        summary.append(row)

    # Free the 3D-CLIP weights before the CLEAR projection (mirrors repo scripts).
    free_3d_clip()

    # --- openai: CLEAR projection into text-embedding-3-large space ---
    if "openai" in modes:
        t0 = time.time()
        try:
            llm_repr = get_or_compute_llm_repr(ds, "openai", img)
            pos, neg = _openai_label_emb(prompts, is_18)
        except (OpenAIUnavailable, FileNotFoundError) as e:
            print(f"[skip openai/{ds}] {e}")
        else:
            probs = softmax_pos_neg(llm_repr, pos, neg)
            row = _save(ds, "openai", probs, y, labels, len(df))
            row["seconds"] = round(time.time() - t0, 1)
            summary.append(row)


def write_summary(summary: list[dict]) -> None:
    if not summary:
        print("[summary] nothing to write")
        return
    df = pd.DataFrame(summary)[
        ["model", "dataset", "mode", "n_volumes", "n_labels", "mean_auc", "seconds"]
    ]
    csv = RESULTS / "summary.csv"
    if csv.exists():
        old = pd.read_csv(csv)
        df = (pd.concat([old, df], ignore_index=True)
                .drop_duplicates(subset=["model", "dataset", "mode"], keep="last"))
    df = df.sort_values(["dataset", "mode"]).reset_index(drop=True)
    df.to_csv(csv, index=False)

    # Markdown view for quick eyeballing.
    md = ["# Ours (CLEAR-3D) — exp1 zero-shot (plain + openai)\n",
          f"Config: `{CLIP3D_BEST}` (run_tag=`{RUN_TAG}`)\n",
          "| dataset | mode | n | labels | mean AUC |",
          "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['dataset']} | {r['mode']} | {int(r['n_volumes'])} "
                  f"| {int(r['n_labels'])} | {r['mean_auc']:.4f} |")
    (RESULTS / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\n=== Summary → {csv} ===")
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(SPECS),
                    choices=list(SPECS) + PMBB_DATASETS)
    ap.add_argument("--modes", nargs="*", default=["plain", "openai"],
                    choices=["plain", "openai"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    print(f"Model: {MODEL} / {CLIP3D_BEST} (run_tag={RUN_TAG})")
    print(f"Datasets: {args.datasets}  Modes: {args.modes}")
    summary: list[dict] = []
    for ds in args.datasets:
        run_dataset(ds, args.modes, args.batch_size, args.num_workers, summary)
    write_summary(summary)


if __name__ == "__main__":
    main()

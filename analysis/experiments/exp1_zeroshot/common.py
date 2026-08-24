#!/usr/bin/env python3
"""
CLEAR_CT_EXPS / exp1_zeroshot — shared baseline utilities.

Every baseline (COLIPRI, CT-CLIP, Merlin, f-VLM) is evaluated zero-shot on the
SAME standardized input as our model: the repo's preprocessed h5 volumes
(``ct_volumes``, ``(N, 160, 224, 224)`` uint8). This is a deliberate
*controlled* comparison — all models see identical input voxels, so AUC
differences reflect the pretrained representation, not preprocessing. The only
per-baseline freedom is (a) resizing our volume to that model's native input
grid and (b) the model's own intensity normalization.

This module centralizes everything that is NOT model-specific:
  * label/volume alignment (reuses ``clear3d.data.load_split`` so the y-matrix
    and volume order are byte-for-byte the same as ours/),
  * the exact uint8 -> HU inverse of the repo preprocessing,
  * a lazy h5 volume Dataset that applies a model-specific transform,
  * AUC + results/summary writers in ours/'s JSON + CSV/MD schema.

A baseline driver therefore only implements: load model, image transform to its
grid, encode text, encode image, score. Scoring reuses
``clear3d.metrics.softmax_pos_neg`` — the identical pos/neg softmax our model
uses — so the comparison is apples-to-apples.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

# --- shared CLEAR-3D package (label alignment, metrics, dataset paths) -------
REPO = os.environ.setdefault("CLEAR3D_REPO", "/path/to/ACT/analysis")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from clear3d.data import load_split                                   # noqa: E402
from clear3d.metrics import mean_auc, per_label_auc, softmax_pos_neg  # noqa: E402
from clear3d.paths import (                                           # noqa: E402
    DATASETS, LABELS_18,
    LABELS_RSNA2023_TRAUMA, LABELS_RSNA2023_TRAUMA_PROMPTS,
)

# ----------------------------------------------------------------------------
# uint8 <-> HU.  The repo preprocessing (clip_3d_ct/run_preprocess.py) clips HU
# to [-1000, 1000] then encodes  uint8 = (clip(HU)/1000 + 1)/2 * 255.  Hence the
# exact inverse below (modulo 8-bit quantization).  Volumes are already padded
# with -1000 HU (=> uint8 0), so reconstructed HU is ready for any baseline's
# own window/normalize step.
# ----------------------------------------------------------------------------
HU_LO, HU_HI = -1000.0, 1000.0


def uint8_to_hu(vol_u8: np.ndarray) -> np.ndarray:
    return vol_u8.astype(np.float32) / 255.0 * (HU_HI - HU_LO) + HU_LO


# dataset -> (label columns, prompt strings, anatomical region).
SPECS = {
    "ctrate_test":   dict(labels=LABELS_18,
                          prompts=[l.lower() for l in LABELS_18], region="chest"),
    "radchest":      dict(labels=LABELS_18,
                          prompts=[l.lower() for l in LABELS_18], region="chest"),
    "rsna2023_test": dict(labels=LABELS_RSNA2023_TRAUMA,
                          prompts=list(LABELS_RSNA2023_TRAUMA_PROMPTS), region="abdomen"),
    # PMBB classification pools. Listed here so they are valid --datasets choices
    # for every runner and so region-aware baselines (merlin) work; the actual
    # labels/prompts are loaded from the phrase-mined CSV in load_dataset (the
    # `labels` below are NOT read for pmbb — load_dataset branches first).
    "pmbb_chest_test": dict(labels=LABELS_18, prompts=[l.lower() for l in LABELS_18],
                            region="chest"),
    "pmbb_abd_test":   dict(labels=[], prompts=[], region="abdomen"),
    # Scaled exp1 pools (supersede the 2,489 pilot): chest NON-contrast (~9k),
    # abd CONTRAST/Merlin-style (~14k). Labels come from load_dataset's pmbb
    # branch (mined CSV header), not the `labels` field here.
    "pmbb_chest_nc":   dict(labels=LABELS_18, prompts=[l.lower() for l in LABELS_18],
                            region="chest"),
    "pmbb_abd_ce":     dict(labels=[], prompts=[], region="abdomen"),
    # INSPECT CTPA + EHR phecode phenotypes (221 phecodes, 2,612 test volumes).
    # labels/prompts come from load_dataset's inspect_pheno branch (the 221
    # phecode_str names), NOT the empty `labels` here. region=chest (CTPA = thorax).
    "inspect_pheno":   dict(labels=[], prompts=[], region="chest"),
}


# Phrase-mined PMBB finding labels (see pmbb_labels/). The CSV header IS the
# label list (single source of truth); prompts are the lowercased names.
PMBB_LABELS_DIR = Path(__file__).resolve().parent / "pmbb_labels" / "labels"
INSPECT_PHENO_DIR = Path(__file__).resolve().parent / "inspect_pheno"


def load_inspect_pheno():
    """(df, h5_idx, y, labels, prompts) for the INSPECT phenotype eval — the exact
    2,612 test volumes × 221 phecodes ours scored (built by inspect_pheno/build_assets.py).
    df carries VolumeName + native NIfTI Path; h5_idx is a row index (native runners
    ignore it). labels = the 221 phecode_str names; prompts = lowercased names."""
    man = pd.read_csv(INSPECT_PHENO_DIR / "manifest.csv")
    y = np.load(INSPECT_PHENO_DIR / "y_221.npy").astype(np.int64)
    ph = pd.read_csv(INSPECT_PHENO_DIR / "phecodes.csv", dtype=str)
    labels = ph["phecode_str"].tolist()
    if len(labels) != y.shape[1] or len(man) != y.shape[0]:
        raise ValueError(f"inspect_pheno asset mismatch: man={len(man)} y={y.shape} "
                         f"labels={len(labels)}")
    return man, man["h5_idx"].values.astype(np.int64), y, labels, [l.lower() for l in labels]


def load_pmbb_labels(name: str, df: pd.DataFrame):
    """(y, labels, prompts) for a PMBB classification pool, joined to df order by
    VolumeName. Raises if the join is incomplete (mis-aligned manifest)."""
    lab = pd.read_csv(PMBB_LABELS_DIR / f"{name}_labels.csv")
    labels = [c for c in lab.columns if c != "VolumeName"]
    merged = df[["VolumeName"]].merge(lab, on="VolumeName", how="left")
    if merged[labels].isna().any().any():
        n = int(merged[labels].isna().any(axis=1).sum())
        raise ValueError(f"{name}: {n} volumes missing mined labels (join mismatch)")
    y = merged[labels].values.astype(np.int64)
    return y, labels, [l.lower() for l in labels]


def load_dataset(name: str):
    """(df, h5_indices, y, labels, prompts) — same order/labels ours/ uses.

    PMBB pools: if a mined label file exists (exp1 classification), return its
    labels; otherwise (exp2 retrieval) return empty y/labels so the 2D retrieval
    runner, which uses only h5_idx, still works."""
    if name == "inspect_pheno":
        return load_inspect_pheno()
    df = load_split(name)
    h5_idx = df["h5_idx"].values.astype(np.int64)
    if name.startswith("pmbb_"):
        if (PMBB_LABELS_DIR / f"{name}_labels.csv").exists():
            y, labels, prompts = load_pmbb_labels(name, df)
            return df, h5_idx, y, labels, prompts
        return df, h5_idx, np.empty((len(df), 0), dtype=np.int64), [], []
    spec = SPECS[name]
    y = df[spec["labels"]].values.astype(np.int64)
    return df, h5_idx, y, spec["labels"], spec["prompts"]


def h5_path(name: str) -> str:
    return DATASETS[name]["h5"]


class H5Volumes(torch.utils.data.Dataset):
    """Lazy reader of our preprocessed h5; applies ``transform(uint8_vol)``.

    ``transform`` maps a ``(160, 224, 224)`` uint8 array to that baseline's
    input tensor (e.g. ``(1, S, S, S)`` float). The h5 file is opened once per
    worker (h5py handles are not fork-safe).
    """

    def __init__(self, name: str, h5_indices: np.ndarray, transform):
        self.path = h5_path(name)
        self.h5_indices = np.asarray(h5_indices, dtype=np.int64)
        self.transform = transform
        self._d = None

    def _ensure_open(self):
        if self._d is None:
            self._d = h5py.File(self.path, "r")["ct_volumes"]

    def __len__(self):
        return len(self.h5_indices)

    def __getitem__(self, i):
        self._ensure_open()
        vol = np.asarray(self._d[int(self.h5_indices[i])])   # (160,224,224) uint8
        return self.transform(vol), i


# ----------------------------------------------------------------------------
# Results writers — identical schema to ours/run_zeroshot.py so a later
# cross-model table can just concat every model's summary.csv.
# ----------------------------------------------------------------------------
def cached_imgfeat(results_dir, dataset: str, mode: str, n: int, encode_fn):
    """Cache/reuse image embeddings (the expensive encode) per (dataset, mode) at
    ``<results_dir>/<dataset>__<mode>__imgfeat.npy``. Returns the (n, dim) features;
    re-runs (e.g. different prompts/scoring) reuse them. Cache is regenerated only if
    it holds fewer than n volumes."""
    results_dir = Path(results_dir)
    p = results_dir / f"{dataset}__{mode}__imgfeat.npy"
    if p.exists():
        arr = np.load(p)
        if arr.shape[0] >= n:
            print(f"[cache] reuse {p.name} ({arr.shape[0]} feats, dim {arr.shape[1]})")
            return arr[:n]
    arr = encode_fn()
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)
    print(f"[cache] saved {p.name} {tuple(arr.shape)}")
    return arr


def save_run(model: str, config: str, dataset: str, mode: str,
             probs: np.ndarray, y: np.ndarray, labels: list[str],
             n: int, results_dir: Path, seconds: float, extra: dict | None = None) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    auc = per_label_auc(probs, y, labels)
    m = mean_auc(auc)
    np.save(results_dir / f"{dataset}__{mode}__probs.npy", probs)
    payload = {
        "model": model, "config": config, "dataset": dataset, "mode": mode,
        "n_volumes": int(n), "n_labels": len(labels), "labels": labels,
        "mean_auc": m, "per_label_auc": auc,
    }
    if extra:
        payload.update(extra)
    (results_dir / f"{dataset}__{mode}__results.json").write_text(json.dumps(payload, indent=2))
    print(f"[{model}/{dataset}/{mode}] mean AUC = {m:.4f}  (n={n}, {len(labels)} labels)")
    return {"model": model, "dataset": dataset, "mode": mode,
            "n_volumes": int(n), "n_labels": len(labels),
            "mean_auc": m, "seconds": round(seconds, 1)}


def write_summary(rows: list[dict], results_dir: Path, title: str) -> None:
    if not rows:
        print("[summary] nothing to write")
        return
    results_dir = Path(results_dir)
    df = pd.DataFrame(rows)[
        ["model", "dataset", "mode", "n_volumes", "n_labels", "mean_auc", "seconds"]
    ]
    csv = results_dir / "summary.csv"
    if csv.exists():
        old = pd.read_csv(csv)
        df = (pd.concat([old, df], ignore_index=True)
                .drop_duplicates(subset=["model", "dataset", "mode"], keep="last"))
    df = df.sort_values(["dataset", "mode"]).reset_index(drop=True)
    df.to_csv(csv, index=False)
    md = [f"# {title}\n", "| dataset | mode | n | labels | mean AUC |",
          "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['dataset']} | {r['mode']} | {int(r['n_volumes'])} "
                  f"| {int(r['n_labels'])} | {r['mean_auc']:.4f} |")
    (results_dir / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\n=== Summary → {csv} ===")
    print(df.to_string(index=False))

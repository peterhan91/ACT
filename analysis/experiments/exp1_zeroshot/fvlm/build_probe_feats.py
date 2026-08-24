#!/usr/bin/env python3
"""
Assemble the 3 per-volume f-VLM probe features from the per-organ arrays dumped by
extract_feats.py (--merge output). Rows stay in C.load_dataset order, so the .npy
drop straight into perseed_probe.py --feat / --feat_train.

  concat1024  (N,1024)  4 organ embeddings (lung,heart,esoph,aorta) concatenated,
                        zero-filled for organs absent/dropped in that volume.
  meanpool256 (N,256)   mean of the present organs' 256-d embeddings.
  vit768      (N,768)   mean of the present organs' raw 3D-ViT pooled tokens.

  python build_probe_feats.py --dataset ctrate_train   # + ctrate_test
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FEAT_DIR = Path(os.environ.get("FVLM_FEAT_DIR", str(HERE / "feats_probe")))


def build(name):
    proj = np.load(FEAT_DIR / f"{name}__organ_proj256.npy")      # (N,4,256)
    vit = np.load(FEAT_DIR / f"{name}__organ_vit768.npy")        # (N,4,768)
    present = np.load(FEAT_DIR / f"{name}__present.npy").astype(np.float32)  # (N,4)
    N = proj.shape[0]
    cnt = present.sum(1, keepdims=True)                           # (N,1) organs present
    denom = np.clip(cnt, 1.0, None)                              # avoid /0 (zero row stays zero)
    concat1024 = proj.reshape(N, -1).astype(np.float32)          # zero-fill already encodes absence
    meanpool256 = (proj * present[:, :, None]).sum(1) / denom
    vit768 = (vit * present[:, :, None]).sum(1) / denom
    for tag, arr in [("fvlm_concat1024", concat1024),
                     ("fvlm_meanpool256", meanpool256.astype(np.float32)),
                     ("fvlm_vit768", vit768.astype(np.float32))]:
        out = FEAT_DIR / f"{name}__{tag}.npy"
        np.save(out, arr)
        print(f"  {tag:16} {arr.shape} -> {out.name}")
    nz = int((cnt[:, 0] == 0).sum())
    print(f"[build] {name}: N={N}  rows with 0 organs (all-zero feature)={nz}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["ctrate_train", "ctrate_test", "radchest", "pmbb_chest_nc"])
    build(ap.parse_args().dataset)


if __name__ == "__main__":
    main()

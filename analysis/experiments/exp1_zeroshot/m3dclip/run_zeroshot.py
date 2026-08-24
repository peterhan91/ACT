#!/usr/bin/env python3
"""
M3D-CLIP (GoodBaiBai88/M3D-CLIP) — exp1 zero-shot baseline.

General 3D CT CLIP (3D ViT + BERT, 768-d). Input (B,1,32,256,256) in [0,1].

Two input protocols (``--source``):
  * h5     — strict h5-only controlled input: our (160,224,224) uint8/255 ->
             trilinear resize to (32,256,256). (mode "plain")
  * native — the FAITHFUL pipeline: raw scan through M3D's OWN preprocessing
             (Data/process/m3d_cap_data_prepare.py): per-volume min-max -> [0,1]
             then CropForeground + Resize([32,256,256]). (mode "native")

Reuses the `ctclip` env (transformers + monai). Scoring = clear3d.metrics.softmax_pos_neg.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
import common as C   # noqa: E402
import native as NV  # noqa: E402

from transformers import AutoModel, AutoTokenizer  # noqa: E402
from monai.data import MetaTensor  # noqa: E402
from monai.transforms import (  # noqa: E402
    Compose, Orientation, CropForeground, ScaleIntensity, Resize,
)

MODEL = "m3dclip"
CONFIG = "GoodBaiBai88/M3D-CLIP"
RESULTS = HERE / "results"
DEVICE = "cuda"
TARGET = (32, 256, 256)


def m3d_transform(vol_u8: np.ndarray) -> torch.Tensor:
    """h5: (160,224,224) uint8 -> (1,32,256,256) float in [0,1]."""
    x = vol_u8.astype(np.float32) / 255.0
    t = torch.from_numpy(x)[None, None]
    t = F.interpolate(t, size=TARGET, mode="trilinear", align_corners=False)
    return t[0]


# M3D's OWN preprocessing (Data/process/m3d_cap_data_prepare.py), in its EXACT
# order: per-volume min-max -> [0,1] FIRST, THEN Compose([CropForeground(),
# Resize([32,256,256])]). M3D's CropForeground uses MONAI's default
# select_fn=is_positive (x>0): its source is 8-bit slice stacks whose background
# is a true 0, so x>0 strips the black border. We replicate faithfully — min-max
# maps air (the volume min, ~-1000 HU) to exactly 0, so x>0 on the normalized
# volume strips the same air background (lungs are interior, >0, so kept). No HU
# clip (M3D doesn't clip — bone sets the max, soft tissue is compressed).
# Output axis order (C,S,A,R) matches the h5/plain path so the two are comparable.
M3D_NATIVE_TF = Compose([
    Orientation(axcodes="RAS"),                               # (1,R,A,S)
    ScaleIntensity(minv=0.0, maxv=1.0),                       # per-volume min-max -> [0,1]  (M3D does this FIRST)
    CropForeground(select_fn=lambda x: x > 0, margin=0),      # M3D default is_positive, on the normalized volume
    Resize(spatial_size=(256, 256, 32), mode="trilinear"),    # on (R,A,S) grid
])


def m3d_native_transform(item: dict) -> torch.Tensor:
    """native: raw HU scan -> M3D's CropForeground+min-max+resize -> (1,32,256,256)."""
    hu, aff = NV.load_hu_affine(item)
    mt = MetaTensor(torch.from_numpy(hu)[None].float(),
                    affine=torch.as_tensor(aff, dtype=torch.float32))
    t = M3D_NATIVE_TF(mt).as_tensor().float()                 # (1,256,256,32) = (C,R,A,S)
    return t.permute(0, 3, 2, 1).contiguous()                 # -> (1,32,256,256) = (C,S,A,R)


@torch.no_grad()
def encode_text(model, tok, prompts: list[str]) -> np.ndarray:
    t = tok(list(prompts), max_length=512, truncation=True,
            padding="max_length", return_tensors="pt")
    emb = model.encode_text(t["input_ids"].to(DEVICE),
                            t["attention_mask"].to(DEVICE))[:, 0]
    return emb.float().cpu().numpy()


@torch.no_grad()
def encode_images(model, ds, batch_size, num_workers) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True)
    out = []
    for batch, _ in tqdm(loader, desc="encode", unit="batch"):
        emb = model.encode_image(batch.to(DEVICE).float())[:, 0]
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def run_dataset(model, tok, name, source, batch_size, num_workers, limit, summary):
    df, h5_idx, y, labels, prompts = C.load_dataset(name)
    n = len(h5_idx)
    if limit:
        n = min(limit, n)
    y = y[:n]
    if source == "h5":
        ds = C.H5Volumes(name, h5_idx[:n], m3d_transform); mode = "plain"
    else:
        _, items = NV.native_index(name)
        ds = NV.NativeVolumes(items[:n], m3d_native_transform); mode = "native"
    print(f"\n===== {name} [{source}]: {n} vols × {len(labels)} labels =====")
    t0 = time.time()
    pos = encode_text(model, tok, [p for p in prompts])
    neg = encode_text(model, tok, [f"no {p}" for p in prompts])
    img = C.cached_imgfeat(RESULTS, name, mode, n,
                           lambda: encode_images(model, ds, batch_size, num_workers))
    probs = C.softmax_pos_neg(img, pos, neg)
    row = C.save_run(MODEL, CONFIG, name, mode, probs, y, labels,
                     n, RESULTS, time.time() - t0, extra={"input_source": source})
    summary.append(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*",
                    default=["ctrate_test", "radchest", "rsna2023_test"],
                    choices=list(C.SPECS))
    ap.add_argument("--source", default="native", choices=["h5", "native"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print(f"Model: {MODEL} / {CONFIG}  |  datasets={args.datasets}  source={args.source}")
    model = AutoModel.from_pretrained(CONFIG, trust_remote_code=True).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(CONFIG, model_max_length=512,
                                        padding_side="right", use_fast=False)
    summary = []
    for ds in args.datasets:
        run_dataset(model, tok, ds, args.source, args.batch_size, args.num_workers, args.limit, summary)
    C.write_summary(summary, RESULTS, f"M3D-CLIP — exp1 zero-shot ({args.source})")


if __name__ == "__main__":
    main()

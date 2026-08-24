#!/usr/bin/env python3
"""
Merlin (stanfordmimi/Merlin) — exp1 zero-shot baseline.

3D abdominal CT VLM (I3-ResNet152 + Clinical-Longformer, 512-d). Natural fit =
rsna2023_test (abdomen); chest sets are OOD reference. Merlin's paper reports on
abdominal phenotype/disease tasks, not these chest findings, so there is no
published zero-shot number to match here — this is a faithful-preprocessing OOD
reference, not a reproduction target.

Two input protocols (``--source``):
  * h5     — strict h5-only controlled input (our (160,224,224) uint8/255 ->
             transpose (224,224,160); == Merlin's ScaleIntensityRange[-1000,1000]
             ->[0,1] target on its RAS grid, no resize).
  * native — the FAITHFUL pipeline: raw scan through Merlin's OWN MONAI
             ``merlin.data.monai_transforms.ImageTransforms`` (LoadImage -> RAS ->
             Spacing 1.5/1.5/3 -> ScaleIntensityRange[-1000,1000]->[0,1] ->
             Pad/Crop (224,224,160)). Raw exists for ctrate_test + rsna2023_test.

Scoring = clear3d.metrics.softmax_pos_neg (image-text cosine, pos vs "no" prompt).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
import common as C   # noqa: E402
import native as NV  # noqa: E402

from merlin import Merlin  # noqa: E402
from monai.data import MetaTensor  # noqa: E402
from monai.transforms import (  # noqa: E402
    Compose, Orientation, Spacing, ScaleIntensityRange, SpatialPad, CenterSpatialCrop,
)

MODEL = "merlin"
CONFIG = "stanfordmimi/Merlin"
RESULTS = HERE / "results"
DEVICE = "cuda"

# Merlin's own MONAI geometry+intensity (array form; == merlin.data.monai_transforms
# ImageTransforms minus LoadImage/EnsureChannelFirst/ToTensor — verified cos=1.0).
MERLIN_ARRAY_TF = Compose([
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(1.5, 1.5, 3.0), mode="bilinear"),
    ScaleIntensityRange(a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
    SpatialPad(spatial_size=[224, 224, 160]),
    CenterSpatialCrop(roi_size=[224, 224, 160]),
])


def merlin_transform(vol_u8: np.ndarray) -> torch.Tensor:
    """h5: (160,224,224) uint8 -> (1,224,224,160) float in [0,1], (R,A,S) order."""
    x = (vol_u8.astype(np.float32) / 255.0).transpose(2, 1, 0).copy()
    return torch.from_numpy(x)[None]


def merlin_native_transform(item: dict) -> torch.Tensor:
    """native: raw scan (nii or radchest .npz) -> Merlin's MONAI pipeline -> (1,224,224,160)."""
    hu, aff = NV.load_hu_affine(item)
    mt = MetaTensor(torch.from_numpy(hu)[None].float(),
                    affine=torch.as_tensor(aff, dtype=torch.float32))
    return MERLIN_ARRAY_TF(mt).as_tensor().float()


@torch.no_grad()
def encode_text(model, prompts: list[str]) -> np.ndarray:
    emb = model.model.encode_text(list(prompts))
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.float().cpu().numpy()


@torch.no_grad()
def encode_images(model, ds, batch_size, num_workers) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True)
    out = []
    for batch, _ in tqdm(loader, desc="encode", unit="batch"):
        feat = model.model.encode_image(batch.to(DEVICE).float())
        emb = feat[0] if isinstance(feat, (tuple, list)) else feat
        emb = emb / emb.norm(dim=-1, keepdim=True)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def run_dataset(model, name, source, batch_size, num_workers, limit, summary):
    df, h5_idx, y, labels, prompts = C.load_dataset(name)
    n = len(h5_idx)
    if limit:
        n = min(limit, n)
    y = y[:n]
    region = C.SPECS[name]["region"]
    tag = "in-region(abdomen)" if region == "abdomen" else "OOD-ref(chest)"
    if source == "h5":
        ds = C.H5Volumes(name, h5_idx[:n], merlin_transform); mode = "plain"
    else:
        _, items = NV.native_index(name)
        ds = NV.NativeVolumes(items[:n], merlin_native_transform); mode = "native"
    print(f"\n===== {name} [{source}/{tag}]: {n} vols × {len(labels)} labels =====")
    t0 = time.time()
    pos = encode_text(model, [p for p in prompts])
    neg = encode_text(model, [f"no {p}" for p in prompts])
    img = C.cached_imgfeat(RESULTS, name, mode, n,
                           lambda: encode_images(model, ds, batch_size, num_workers))
    probs = C.softmax_pos_neg(img, pos, neg)
    row = C.save_run(MODEL, CONFIG, name, mode, probs, y, labels,
                     n, RESULTS, time.time() - t0,
                     extra={"input_source": source, "region_fit": tag})
    summary.append(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*",
                    default=["rsna2023_test", "ctrate_test", "radchest"],
                    choices=list(C.SPECS))
    ap.add_argument("--source", default="native", choices=["h5", "native"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    print(f"Model: {MODEL} / {CONFIG}  |  datasets={args.datasets}  source={args.source}")
    model = Merlin().to(DEVICE).eval()
    summary = []
    for ds in args.datasets:
        run_dataset(model, ds, args.source, args.batch_size, args.num_workers, args.limit, summary)
    C.write_summary(summary, RESULTS, f"Merlin — exp1 zero-shot ({args.source})")


if __name__ == "__main__":
    main()

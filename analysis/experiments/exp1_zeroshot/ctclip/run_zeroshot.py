#!/usr/bin/env python3
"""
CT-CLIP (CT-RATE authors, CT-CLIP_v2) — exp1 zero-shot baseline.

Shares OUR exact 18 CT-RATE pathologies + present/absent prompt convention.
Two input protocols (``--source``):
  * h5     — strict h5-only controlled input (our (160,224,224) uint8 -> recon HU
             -> /1000 [-1,1] -> trilinear resize to (1,240,480,480)).
  * native — the FAITHFUL paper pipeline: raw CT-RATE NIfTI through CT-CLIP's OWN
             ``data_inference_nii.nii_img_to_tensor`` (resample to 0.75/0.75/1.5 mm,
             HU clip [-1000,1000], /1000, center crop/pad to (480,480,240), pad=-1).
             CT-RATE valid_fixed is already in HU, so we neutralize the metadata
             RescaleSlope/Intercept (else they double-subtract). ctrate_test only
             (the loader needs the CT-RATE metadata). NOTE: we evaluate on our
             ctrate_test (2489 vols); the paper reports on its own 1304-vol valid set.

Prompts/scoring are CT-CLIP's own (scripts/zero_shot.py): per finding,
softmax over ["{p} is present.", "{p} is not present."], take the present prob —
i.e. clear3d.metrics.softmax_pos_neg.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
import common as C   # noqa: E402
import native as NV  # noqa: E402
from clear3d.paths import DATASETS  # noqa: E402

# torch>=2.6 defaults weights_only=True, which can't unpickle the v2 checkpoint.
_ORIG_LOAD = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _ORIG_LOAD(*a, **k)
torch.load = _patched_load

# CT-CLIP's own NIfTI preprocessing (for native mode).
sys.path.insert(0, str(HERE / "CT-CLIP" / "scripts"))
import data_inference_nii as _dn             # noqa: E402

from transformer_maskgit import CTViT          # noqa: E402
from transformers import BertTokenizer, BertModel  # noqa: E402
from ct_clip import CTCLIP                      # noqa: E402

MODEL = "ctclip"
CONFIG = "CT-CLIP_v2"
WEIGHTS = HERE / "weights/models/CT-CLIP-Related/CT-CLIP_v2.pt"
RESULTS = HERE / "results"
DEVICE = "cuda"
TARGET = (240, 480, 480)


def ctclip_transform(vol_u8: np.ndarray) -> torch.Tensor:
    """h5: (160,224,224) uint8 -> (1,240,480,480) float in [-1,1]."""
    hu = C.uint8_to_hu(vol_u8)
    x = np.clip(hu, -1000, 1000) / 1000.0
    t = torch.from_numpy(x)[None, None]
    t = F.interpolate(t, size=TARGET, mode="trilinear", align_corners=False)
    return t[0]


def ctclip_native_transform(item: dict) -> torch.Tensor:
    """native: any raw scan -> (HU, affine) -> temp NIfTI -> CT-CLIP's own
    nii_img_to_tensor -> (1,240,480,480). Spacing comes from the affine; valid_fixed /
    radchest .npz / RSNA are already in HU, so slope/intercept are identity. Routing
    through a temp NIfTI reuses CT-CLIP's exact loader for every dataset (no reimpl)."""
    hu, aff = NV.load_hu_affine(item)
    fd, tmp = tempfile.mkstemp(suffix=".nii.gz"); os.close(fd)
    try:
        nib.save(nib.Nifti1Image(hu, aff), tmp)
        # voxel spacing = affine column norms — robust to oblique/rotated affines
        # (some PMBB scans have a ~0 diagonal with spacing in the off-diagonal, which
        # made abs(diag) -> 0 -> CT-CLIP resize to H=0 -> crash). == abs(diag) for
        # axis-aligned scans (CT-RATE/RAD-ChestCT/RSNA), so backward-compatible.
        sx, sy, sz = (float(np.linalg.norm(aff[:3, i])) for i in range(3))
        meta = pd.DataFrame([{"VolumeName": os.path.basename(tmp),
                              "XYSpacing": f"[{sx}, {sy}]", "ZSpacing": sz,
                              "RescaleSlope": 1.0, "RescaleIntercept": 0.0}])
        return _dn.CTReportDatasetinfer.nii_img_to_tensor(None, tmp, meta).float()
    finally:
        os.remove(tmp)


def build_model():
    tok = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized",
                                        do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tok))
    image_encoder = CTViT(dim=512, codebook_size=8192, image_size=480, patch_size=20,
                          temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
                          dim_head=32, heads=8)
    clip = CTCLIP(image_encoder=image_encoder, text_encoder=text_encoder,
                  dim_image=294912, dim_text=768, dim_latent=512,
                  extra_latent_projection=False, use_mlm=False,
                  downsample_image_embeds=False, use_all_token_embeds=False)
    state = torch.load(str(WEIGHTS), map_location="cpu")
    missing, unexpected = clip.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("position_ids")]
    assert not unexpected, f"unexpected non-benign keys in checkpoint: {unexpected[:8]}"
    print(f"[ctclip load] OK (dropped position_ids; {len(missing)} reinit keys e.g. {missing[:2]})")
    return clip.to(DEVICE).eval(), tok


@torch.no_grad()
def encode_text(clip, tok, prompts: list[str]) -> np.ndarray:
    t = tok(list(prompts), return_tensors="pt", padding="max_length",
            truncation=True, max_length=512).to(DEVICE)
    enc = clip.text_transformer(t.input_ids, attention_mask=t.attention_mask)[0]
    emb = clip.to_text_latent(enc[:, 0, :])
    return F.normalize(emb, dim=-1).float().cpu().numpy()


@torch.no_grad()
def encode_images(clip, ds, batch_size, num_workers) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        shuffle=False, pin_memory=True)
    out = []
    for batch, _ in tqdm(loader, desc="encode", unit="batch"):
        enc = clip.visual_transformer(batch.to(DEVICE).float(), return_encoded_tokens=True)
        enc = torch.mean(enc, dim=1).reshape(enc.shape[0], -1)
        emb = F.normalize(clip.to_visual_latent(enc), dim=-1)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def run_dataset(clip, tok, name, source, batch_size, num_workers, limit, summary):
    df, h5_idx, y, labels, prompts = C.load_dataset(name)
    n = len(h5_idx)
    if limit:
        n = min(limit, n)
    y = y[:n]
    if source == "h5":
        ds = C.H5Volumes(name, h5_idx[:n], ctclip_transform); mode = "plain"
    else:
        _, items = NV.native_index(name)
        ds = NV.NativeVolumes(items[:n], ctclip_native_transform); mode = "native"
    print(f"\n===== {name} [{source}]: {n} vols × {len(labels)} labels =====")
    t0 = time.time()
    pos = encode_text(clip, tok, [f"{p} is present." for p in prompts])
    neg = encode_text(clip, tok, [f"{p} is not present." for p in prompts])
    img = C.cached_imgfeat(RESULTS, name, mode, n,
                           lambda: encode_images(clip, ds, batch_size, num_workers))
    probs = C.softmax_pos_neg(img, pos, neg)
    row = C.save_run(MODEL, CONFIG, name, mode, probs, y, labels,
                     n, RESULTS, time.time() - t0,
                     extra={"input_source": source, "prompts": "present/absent"})
    summary.append(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*",
                    default=["ctrate_test", "radchest", "rsna2023_test"],
                    choices=list(C.SPECS))
    ap.add_argument("--source", default="native", choices=["h5", "native"])
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    assert WEIGHTS.exists(), f"missing CT-CLIP weights at {WEIGHTS}"
    print(f"Model: {MODEL} / {CONFIG}  |  datasets={args.datasets}  source={args.source}")
    clip, tok = build_model()
    summary = []
    for ds in args.datasets:
        run_dataset(clip, tok, ds, args.source, args.batch_size, args.num_workers, args.limit, summary)
    C.write_summary(summary, RESULTS, f"CT-CLIP_v2 — exp1 zero-shot ({args.source})")


if __name__ == "__main__":
    main()

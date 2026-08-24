#!/usr/bin/env python3
"""
Dump f-VLM per-organ features for LINEAR PROBING (no scoring). Mirrors
run_zeroshot.run_dataset's official preprocessing + per-organ center-crop + forward,
but captures the embeddings the model computes internally via the opt-in
blip_pretrain.forward_test_win(feat_sink=...) hook (no change to the scoring path).

For each volume, for each of the 4 chest organs (lung/heart/esophagus/aorta) present
in its TotalSegmentator mask, we crop to that organ exactly as scoring does
(center_crop + DivisiblePad, skip_organ=oid) and read back:
  proj256  (256,)  the OFFICIAL organ embedding  = F.normalize(vision_proj(query))
  vit768   (768,)  mean of the raw 3D-ViT tokens for that organ ROI (pre-attention)
Absent/dropped organs are zero-filled; a presence flag records which were captured.

Rows align 1:1 to C.load_dataset(name) -> exactly the order perseed_probe.py expects,
so the assembled feature matrices are directly probe-able against the same y.

Sharded + resumable like gen_masks.py (one block per --shard; --merge stitches blocks):
  python extract_feats.py --dataset ctrate_train --nshards 12 --shard 0   # one block
  python extract_feats.py --dataset ctrate_train --merge                  # blocks -> full
Outputs under FVLM_FEAT_DIR (default feats_probe/):
  <dataset>__shard{i}of{N}__featblock.npz     per-shard (rows lo..hi + index)
  <dataset>__organ_proj256.npy  (N,4,256)  <dataset>__organ_vit768.npy (N,4,768)
  <dataset>__present.npy        (N,4) int8
build_probe_feats.py then assembles concat1024 / meanpool256 / vit768 (N,D).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                 # exp1_zeroshot
import run_zeroshot as RZ                              # noqa: E402  (builds model, preprocess, etc.)

C, NV = RZ.C, RZ.NV
ORGANS, ROI, DEVICE = RZ.ORGANS, RZ.ROI, RZ.DEVICE
FEAT_DIR = Path(os.environ.get("FVLM_FEAT_DIR", str(HERE / "feats_probe")))


@torch.no_grad()
def extract_block(model, name, nshards, shard, limit=0):
    df, _h5, y, labels, _ = C.load_dataset(name)             # SAME order perseed_probe uses
    id_col = "NoteAcc_DEID" if name == "radchest" else "VolumeName"
    vols = df[id_col].astype(str).tolist()
    masks_dir = HERE / (("masks" if name == "ctrate_test" else f"masks_{name}") + RZ.MASKS_SUFFIX)
    _, items = NV.native_index(name)
    n = len(vols) if not limit else min(limit, len(vols))
    if shard >= 0:
        per = math.ceil(n / max(1, nshards)); lo, hi = shard * per, min((shard + 1) * per, n)
    else:
        lo, hi = 0, n
    rows = hi - lo
    proj = np.zeros((rows, len(ORGANS), 256), np.float32)
    vit = np.zeros((rows, len(ORGANS), 768), np.float32)
    present = np.zeros((rows, len(ORGANS)), np.int8)
    print(f"[extract] {name} rows [{lo},{hi}) of {n} -> {FEAT_DIR}", flush=True)
    t0 = time.time(); done = miss = 0
    for r, i in enumerate(range(lo, hi)):
        fname = vols[i] if vols[i].endswith(".nii.gz") else vols[i] + ".nii.gz"
        mpath = masks_dir / fname
        if not mpath.exists():
            miss += 1
            continue
        tmp = None
        try:
            if items[i]["kind"] == "nii":
                img_path = items[i]["path"]
            else:
                hu, aff = NV.load_hu_affine(items[i])
                fd, tmp = tempfile.mkstemp(suffix=".nii.gz"); os.close(fd)
                nib.save(nib.Nifti1Image(hu, aff), tmp); img_path = tmp
            image, mask = RZ.preprocess(img_path, str(mpath))
            image = image[None].to(DEVICE); mask = mask[None].to(DEVICE)        # (1,1,D,H,W)
            pres = [o for o in ORGANS if torch.eq(mask, ORGANS.index(o) + 1).any()]
            whole = {o: torch.eq(mask, ORGANS.index(o) + 1).sum().item() for o in pres}
            for o in pres:                                                       # per-organ crop, as scoring does
                oid = ORGANS.index(o)
                wp, wm = RZ.center_crop(image, torch.eq(mask, oid + 1), ROI)
                wm = wm.float(); wm[wm == 1] = oid + 1
                pad = RZ._PAD({"image": wp[0], "label": wm[0]})
                sink = {"proj256": {}, "query768": {}, "vit768": {}}
                model.forward_test_win(pad["image"][None], pad["label"][None], {}, pres,
                                       {}, {}, whole, skip_organ=oid, feat_sink=sink)
                if o in sink["proj256"]:
                    proj[r, oid] = sink["proj256"][o].numpy()
                    vit[r, oid] = sink["vit768"][o].numpy()
                    present[r, oid] = 1
            done += 1
            if done % 50 == 0:
                print(f"  {i+1}/{n} (done={done}, miss={miss}, {(time.time()-t0)/done:.1f}s/vol)", flush=True)
        except Exception as e:
            print(f"  [FAIL {i+1}/{n}] {vols[i]}: {repr(e)[:120]}", flush=True)
        finally:
            if tmp:
                os.remove(tmp)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out = FEAT_DIR / f"{name}__shard{shard}of{nshards}__featblock.npz"
    np.savez_compressed(out, proj256=proj, vit768=vit, present=present,
                        lo=np.int64(lo), hi=np.int64(hi), n=np.int64(n))
    print(f"[extract] {name} block [{lo},{hi}) captured={done} missing_mask={miss} -> {out.name} "
          f"({time.time()-t0:.0f}s)", flush=True)


def merge(name):
    n = None; blocks = sorted(FEAT_DIR.glob(f"{name}__shard*__featblock.npz"))
    assert blocks, f"no shard blocks for {name} in {FEAT_DIR}"
    for b in blocks:
        if n is None:
            n = int(np.load(b)["n"])
            proj = np.zeros((n, len(ORGANS), 256), np.float32)
            vit = np.zeros((n, len(ORGANS), 768), np.float32)
            present = np.zeros((n, len(ORGANS)), np.int8)
            filled = np.zeros(n, bool)
    for b in blocks:
        d = np.load(b); lo, hi = int(d["lo"]), int(d["hi"])
        proj[lo:hi] = d["proj256"]; vit[lo:hi] = d["vit768"]; present[lo:hi] = d["present"]
        filled[lo:hi] = True
    assert filled.all(), f"{name}: {int((~filled).sum())} rows not covered by any shard block"
    np.save(FEAT_DIR / f"{name}__organ_proj256.npy", proj)
    np.save(FEAT_DIR / f"{name}__organ_vit768.npy", vit)
    np.save(FEAT_DIR / f"{name}__present.npy", present)
    cov = present.sum(0)
    print(f"[merge] {name}: N={n}  organ coverage (lung/heart/esoph/aorta)={cov.tolist()}  "
          f"any-organ={int((present.sum(1) > 0).sum())}/{n} -> {FEAT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["ctrate_train", "ctrate_test", "radchest", "pmbb_chest_nc"])
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="stitch shard blocks -> full feature arrays")
    args = ap.parse_args()
    if args.merge:
        merge(args.dataset); return
    model = RZ.build_fvlm()
    extract_block(model, args.dataset, args.nshards, args.shard, args.limit)


if __name__ == "__main__":
    main()

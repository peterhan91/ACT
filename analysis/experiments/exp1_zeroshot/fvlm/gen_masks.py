#!/usr/bin/env python3
"""Generate f-VLM's 4-organ masks via TotalSegmentator (total/fast) merged to
{1:lung, 2:heart, 3:esophagus, 4:aorta} (exactly fvlm/data/resize.py). Dataset-keyed:
  ctrate_test -> masks/          keyed by VolumeName
  radchest    -> masks_radchest/ keyed by NoteAcc_DEID
Loads NIfTI (ctrate) or .npz (radchest) uniformly via native.load_hu_affine. Resumable.
  python gen_masks.py --dataset ctrate_test|radchest [--limit N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # exp1_zeroshot (native.py)
import native as NV                                    # noqa: E402
from totalsegmentator.python_api import totalsegmentator  # noqa: E402

# TS v2 'total' label IDs -> f-VLM organ value (lung lobes->1, heart+atrial_app->2,
# esophagus->3, aorta->4).
TS2ORGAN = {10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 51: 2, 61: 2, 15: 3, 52: 4}
ID_COL = {"ctrate_test": "VolumeName", "ctrate_train": "VolumeName",
          "radchest": "NoteAcc_DEID", "pmbb_chest_nc": "VolumeName"}


def img_of(item):
    """item -> nib image for TS (NIfTI loaded directly; .npz rebuilt from HU+affine)."""
    if item["kind"] == "nii":
        return nib.load(item["path"])
    hu, aff = NV.load_hu_affine(item)
    return nib.Nifti1Image(hu, aff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ctrate_test",
                    choices=["ctrate_test", "ctrate_train", "radchest", "pmbb_chest_nc"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1, help="split dataset into N contiguous blocks")
    ap.add_argument("--shard", type=int, default=-1, help=">=0 selects the i-th block (parallel gen)")
    args = ap.parse_args()
    out = HERE / ("masks" if args.dataset == "ctrate_test" else f"masks_{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)
    df, items = NV.native_index(args.dataset)
    ids = df[ID_COL[args.dataset]].astype(str).tolist()
    n = len(ids) if not args.limit else min(args.limit, len(ids))
    if args.shard >= 0 and args.nshards > 1:
        import math
        per = math.ceil(n / args.nshards)
        lo, hi = args.shard * per, min((args.shard + 1) * per, n)
    else:
        lo, hi = 0, n
    print(f"[{args.dataset}] masks for indices [{lo},{hi}) of {n} -> {out}", flush=True)
    done = fail = skip = 0
    for i in range(lo, hi):
        fid = ids[i]
        fname = fid if fid.endswith(".nii.gz") else fid + ".nii.gz"
        p = out / fname
        if p.exists():
            skip += 1
            continue
        try:
            ts = totalsegmentator(img_of(items[i]), task="total", fast=True, quiet=True)
            arr = np.asanyarray(ts.dataobj).astype(np.int16)
            merged = np.zeros(arr.shape, dtype=np.uint8)
            for tid, org in TS2ORGAN.items():
                merged[arr == tid] = org
            nib.save(nib.Nifti1Image(merged, ts.affine), p)
            done += 1
            if done <= 3 or done % 50 == 0:
                present = sorted(int(v) for v in np.unique(merged) if v)
                print(f"  [{i+1}/{n}] {fid} organs={present} (done={done} skip={skip} fail={fail})",
                      flush=True)
        except Exception as e:
            fail += 1
            print(f"  [FAIL {i+1}/{n}] {fid}: {repr(e)[:120]}", flush=True)
    print(f"MASKS DONE [{args.dataset}] done={done} skip={skip} fail={fail} -> {out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Resume ctrate_train f-VLM mask generation after the 2026-06-30 reboot killed the
original 12-shard run mid-flight. The 5,828 still-missing volumes are scattered
across 126 gap-runs (mostly single-item TotalSeg failures from the original run,
plus one ~1,364-volume tail block the shards never reached) rather than one clean
tail, so sharding the FULL [0,47146) range again (like the original run did) would
dump nearly all the real work onto whichever shard covers the tail block while the
rest just fast-skip. This shards the MISSING list itself for even load, reusing
gen_masks.py's own img_of()/TS2ORGAN/ID_COL so the per-volume logic (and output
path convention) stays byte-identical to the original run.

  python gen_masks_missing_shards.py --nshards N --shard i
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                   # exp1_zeroshot (native.py)
import native as NV                                     # noqa: E402
import gen_masks as GM                                  # noqa: E402  (img_of, TS2ORGAN, ID_COL)

DATASET = "ctrate_train"
OUT = HERE / f"masks_{DATASET}"


def missing_indices():
    df, items = NV.native_index(DATASET)
    ids = df[GM.ID_COL[DATASET]].astype(str).tolist()
    fnames = [fid if fid.endswith(".nii.gz") else fid + ".nii.gz" for fid in ids]
    missing = [i for i, fname in enumerate(fnames) if not (OUT / fname).exists()]
    return items, ids, fnames, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=-1, help=">=0 selects the i-th block of the MISSING list")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items, ids, fnames, missing = missing_indices()
    N = max(1, args.nshards)
    if args.shard >= 0 and N > 1:
        import math
        per = math.ceil(len(missing) / N)
        lo, hi = args.shard * per, min((args.shard + 1) * per, len(missing))
        block = missing[lo:hi]
    else:
        block = missing
    print(f"[{DATASET}] shard {args.shard}/{N}: {len(block)} of {len(missing)} missing volumes "
          f"(dataset total {len(ids)}) -> {OUT}", flush=True)
    done = fail = skip = 0
    for j, i in enumerate(block):
        fid, fname = ids[i], fnames[i]
        p = OUT / fname
        if p.exists():   # another shard/run may have filled it since missing_indices() ran
            skip += 1
            continue
        try:
            # nr_thr_saving defaults to 6 -> each call was silently spawning 6
            # extra multiprocessing workers (observed via `ps --forest`: every
            # shard fanned out to ~6 `spawn_main` children at 35-65% CPU each),
            # which is why running this alongside biomedclip oversubscribed the
            # node (load avg 78 on 72 cores) and stalled it. nr_thr_resamp
            # already defaults to 1. Pin both to 1 so a shard is really 1 process.
            ts = GM.totalsegmentator(GM.img_of(items[i]), task="total", fast=True, quiet=True,
                                     nr_thr_resamp=1, nr_thr_saving=1)
            arr = np.asanyarray(ts.dataobj).astype(np.int16)
            merged = np.zeros(arr.shape, dtype=np.uint8)
            for tid, org in GM.TS2ORGAN.items():
                merged[arr == tid] = org
            nib.save(nib.Nifti1Image(merged, ts.affine), p)
            done += 1
            if done <= 3 or done % 50 == 0:
                present = sorted(int(v) for v in np.unique(merged) if v)
                print(f"  [{j + 1}/{len(block)}] idx={i} {fid} organs={present} "
                      f"(done={done} skip={skip} fail={fail})", flush=True)
        except Exception as e:
            fail += 1
            print(f"  [FAIL {j + 1}/{len(block)}] idx={i} {fid}: {repr(e)[:120]}", flush=True)
    print(f"MASKS DONE [{DATASET} shard {args.shard}/{N}] done={done} skip={skip} fail={fail} -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()

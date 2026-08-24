#!/usr/bin/env python3
"""Full-resolution variant of gen_masks.py — matches the fVLM repo's data/generate_mask.py
(default TotalSegmentator, i.e. fast=False / 1.5mm), as opposed to gen_masks.py's fast=True.
Writes to masks_fullres/ (ctrate) / masks_<dataset>_fullres/ so the existing fast masks are
untouched and we can A/B them. Same {1:lung,2:heart,3:esophagus,4:aorta} merge. Resumable.
  totalseg_venv/bin/python gen_masks_fullres.py --dataset ctrate_test [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import native as NV  # noqa: E402
from totalsegmentator.python_api import totalsegmentator  # noqa: E402

TS2ORGAN = {10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 51: 2, 61: 2, 15: 3, 52: 4}
ID_COL = {"ctrate_test": "VolumeName", "radchest": "NoteAcc_DEID"}
# Only the 4 organs fVLM uses -> TS runs only the sub-models containing them (big speedup);
# masks for these organs are identical to a full run.
ROI_SUBSET = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
              "lung_middle_lobe_right", "lung_lower_lobe_right", "heart", "esophagus", "aorta"]


def img_of(item):
    if item["kind"] == "nii":
        return nib.load(item["path"])
    hu, aff = NV.load_hu_affine(item)
    return nib.Nifti1Image(hu, aff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ctrate_test", choices=["ctrate_test", "radchest"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0, help="this worker's index (0..world-1)")
    ap.add_argument("--world", type=int, default=1, help="number of parallel workers")
    args = ap.parse_args()
    suffix = "_fullres"
    out = HERE / (("masks" if args.dataset == "ctrate_test" else f"masks_{args.dataset}") + suffix)
    out.mkdir(parents=True, exist_ok=True)
    df, items = NV.native_index(args.dataset)
    ids = df[ID_COL[args.dataset]].astype(str).tolist()
    n = len(ids) if not args.limit else min(args.limit, len(ids))
    done = fail = skip = 0
    t0 = time.time()
    for i in range(args.rank, n, args.world):
        fid = ids[i]
        fname = fid if fid.endswith(".nii.gz") else fid + ".nii.gz"
        p = out / fname
        if p.exists():
            skip += 1
            continue
        try:
            # nr_thr_*=1 -> no internal multiprocessing pool (avoids the nnUNet
            # 'ForkAwareLocal' fork-teardown bug under many concurrent workers) and
            # cuts CPU contention so we can run more workers in parallel.
            ts = totalsegmentator(img_of(items[i]), task="total", fast=False,
                                  roi_subset=ROI_SUBSET, quiet=True,
                                  nr_thr_resamp=1, nr_thr_saving=1)
            arr = np.asanyarray(ts.dataobj).astype(np.int16)
            merged = np.zeros(arr.shape, dtype=np.uint8)
            for tid, org in TS2ORGAN.items():
                merged[arr == tid] = org
            nib.save(nib.Nifti1Image(merged, ts.affine), p)
            done += 1
            if done <= 3 or done % 25 == 0:
                present = sorted(int(v) for v in np.unique(merged) if v)
                rate = (time.time() - t0) / done
                print(f"  [{i+1}/{n}] {fid} organs={present} (done={done} skip={skip} "
                      f"fail={fail}) {rate:.1f}s/vol", flush=True)
        except Exception as e:
            fail += 1
            print(f"  [FAIL {i+1}/{n}] {fid}: {repr(e)[:120]}", flush=True)
    print(f"MASKS DONE [{args.dataset}] done={done} skip={skip} fail={fail} "
          f"elapsed={time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()

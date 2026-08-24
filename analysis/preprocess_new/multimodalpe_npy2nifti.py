#!/usr/bin/env python3
"""Convert MultimodalPE .npy volumes to .nii.gz so they flow through the
shared run_preprocess.py pipeline (which expects NIfTI input).

The .npy files are int16, shape (D, H, W) with axes:
  D = inferior -> superior
  H = anterior -> posterior
  W = patient-LEFT -> patient-RIGHT
(verified by side-by-side rendering against ctrate_train.h5 / merlin_train.h5).

We write a NIfTI carrying an affine that describes this orientation in RAS
world coords, so nib.as_closest_canonical produces canonical RAS data shape
(X=L->R, Y=P->A, Z=I->S). Combined with run_preprocess.py's
transpose(2, 1, 0) + slide_b2u=False (axis-1 flip), the final h5 volume has
axes (D=I->S, H=A->P, W=L->R) -- matching CT-RATE/Merlin/INSPECT.

Output filename = sanitized rel path: <subdir>__<idx>.nii.gz (e.g. 2__1880.nii.gz).
Also emits <output_dir>/_paths.csv with one Path column.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

# Voxel (i, j, k) -> RAS (R, A, S):
#   i along D (axis-0): D=I->S means +i -> +S       (S row: [1, 0, 0])
#   j along H (axis-1): H=A->P means +j -> -A       (A row: [0, -1, 0])
#   k along W (axis-2): W=L->R means +k -> +R       (R row: [0, 0, 1])
_AFFINE = np.array([
    [0,  0, 1, 0],   # R
    [0, -1, 0, 0],   # A
    [1,  0, 0, 0],   # S
    [0,  0, 0, 1],
], dtype=np.float64)


def _convert_one(args):
    npy_path, out_path = args
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        return ("SKIP", str(npy_path), str(out_path), None)
    try:
        vol = np.load(npy_path)
        if vol.ndim != 3:
            return ("FAIL", str(npy_path), str(out_path),
                    f"expected 3D, got shape {vol.shape}")
        # Cast to int16 (MultimodalPE files already are; defensive).
        if vol.dtype != np.int16:
            vol = vol.astype(np.int16)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = nib.Nifti1Image(vol, affine=_AFFINE)
        nib.save(img, str(out_path))
        ok = out_path.exists() and out_path.stat().st_size > 0
        return ("OK" if ok else "EMPTY", str(npy_path), str(out_path),
                f"shape={tuple(vol.shape)}")
    except Exception as e:  # noqa: BLE001
        return ("FAIL", str(npy_path), str(out_path),
                f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="MultimodalPE root, e.g. .../multimodalpulmonaryembolismdataset")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write .nii.gz files (one per .npy)")
    ap.add_argument("--paths_csv", required=True,
                    help="Output CSV with one `Path` column listing each .nii.gz")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: convert only the first N .npy files")
    args = ap.parse_args()

    in_root = Path(args.input_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    Path(args.paths_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"[scan] discovering .npy under {in_root}", flush=True)
    npys = sorted(in_root.rglob("*.npy"))
    if args.limit > 0:
        npys = npys[: args.limit]
    print(f"[scan] {len(npys)} .npy files found", flush=True)

    jobs = []
    for p in npys:
        rel = p.relative_to(in_root)
        # 0/1880.npy -> 0__1880.nii.gz
        name = str(rel.with_suffix("")).replace(os.sep, "__") + ".nii.gz"
        jobs.append((p, out_root / name))

    n_ok = n_skip = n_fail = 0
    succeeded = []
    fail_log = Path(args.output_dir) / "_failures.txt"

    print(f"[convert] running with {args.num_workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool, \
            open(fail_log, "w") as flog, \
            tqdm(total=len(jobs), desc="npys") as bar:
        futures = {pool.submit(_convert_one, j): j for j in jobs}
        for fut in as_completed(futures):
            status, src, op, info = fut.result()
            if status == "OK":
                n_ok += 1
                succeeded.append(op)
            elif status == "SKIP":
                n_skip += 1
                succeeded.append(op)
            else:
                n_fail += 1
                flog.write(f"{src}\t{status}\t{info or ''}\n")
                flog.flush()
            bar.update(1)

    print(f"[done] OK={n_ok}  SKIP_EXISTS={n_skip}  FAIL={n_fail}  -> {fail_log}", flush=True)
    succeeded.sort()
    pd.DataFrame({"Path": succeeded}).to_csv(args.paths_csv, index=False)
    print(f"[paths] wrote {len(succeeded)} entries to {args.paths_csv}", flush=True)


if __name__ == "__main__":
    main()

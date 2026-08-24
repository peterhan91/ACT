#!/usr/bin/env python3
"""Raw-scan loading for faithful per-model NATIVE preprocessing.

Each baseline's native run reads the ORIGINAL CT scan and applies that model's OWN
published preprocessing + zero-shot scoring (in each ``<model>/run_zeroshot.py``).
Everything is unified to ``(HU volume, affine)`` so a model transform can build its
input via an array+affine (verified byte-identical to loading the file path, cos=1.0).

Raw sources (all already in HU; order follows ``clear3d.data.load_split`` == common):
  * ctrate_test    — CT-RATE valid_fixed NIfTI (air -1024, bone +3071)
  * rsna2023_test  — RSNA-2023 NIfTI (HU)
  * radchest       — RAD-ChestCT ``.npz['ct']`` (z,y,x) int16, HU-clipped [-1000,1000],
                     0.8 mm isotropic (metadata final_spacing=0.8), DICOM LPS axial
                     (orig_orientation 1,0,0,0,1,0) — same convention as CT-RATE.
"""
from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from pathlib import Path

REPO = os.environ.setdefault("CLEAR3D_REPO", "/path/to/ACT/analysis")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from clear3d.data import load_split  # noqa: E402

CTRATE_ROOT = "/path/to/data/CT_RATE_v2.0/dataset"
RADCHEST_ROOT = "/path/to/ACT/preprocessing/external/radchestct_raw"
RADCHEST_SPACING = 0.8   # mm isotropic (RAD-ChestCT final_spacing)


def ctrate_path(vn: str) -> str:
    s = vn[:-7] if vn.endswith(".nii.gz") else vn
    p = s.split("_")
    split = "valid_fixed" if p[0] == "valid" else "train_fixed"
    return os.path.join(CTRATE_ROOT, split, f"{p[0]}_{p[1]}",
                        f"{p[0]}_{p[1]}_{p[2]}", s + ".nii.gz")


def native_index(dataset: str):
    """(df, items) aligned to load_split order. item = {kind, path}."""
    if dataset == "inspect_pheno":
        # INSPECT phenotype eval: manifest carries the native CTPA NIfTI path per
        # volume (built by inspect_pheno/build_assets.py), already HU-valued.
        man = pd.read_csv(Path(__file__).resolve().parent / "inspect_pheno" / "manifest.csv")
        items = [{"kind": "nii", "path": str(p)} for p in man["Path"]]
        return man, items
    df = load_split(dataset)
    if dataset == "ctrate_test":
        items = [{"kind": "nii", "path": ctrate_path(v)} for v in df["VolumeName"].astype(str)]
    elif dataset == "rsna2023_test":
        items = [{"kind": "nii", "path": str(p).replace(os.environ.get("MANIFEST_PATH_PREFIX", ""), "")} for p in df["Path"]]
    elif dataset == "radchest":
        items = [{"kind": "npz", "path": os.path.join(RADCHEST_ROOT, f"{a}.npz")}
                 for a in df["NoteAcc_DEID"].astype(str)]
    elif dataset.startswith("pmbb_"):
        # PMBB retrieval pools: manifest carries the original full-res NIfTI path
        # (already HU-valued), read by each model's own native transform.
        items = [{"kind": "nii", "path": str(p)} for p in df["Path"]]
    else:
        raise ValueError(f"no raw scans for {dataset} (native unsupported; use h5)")
    return df, items


def _sanitize_affine(aff) -> np.ndarray:
    """Return a decomposable 4x4 affine with plausible CT voxel spacing. Some PMBB
    NIfTIs ship a degenerate affine (all-zero direction row -> singular 3x3, which
    crashes nibabel set_qform / MONAI) or a corrupt slice spacing (e.g. z = 605 mm,
    which makes MONAI Spacing / CT-CLIP resample allocate a ~1.4e9-voxel grid -> a
    single-worker OOM). A clean affine with in-range spacing passes through unchanged;
    a singular one is rebuilt axis-aligned, and any axis spacing outside [0.05, 10] mm
    is clamped to that range with its direction preserved (legit 5 mm thick-slice CT
    is untouched)."""
    SP_LO, SP_HI = 0.05, 10.0     # plausible CT voxel spacing (mm)
    aff = np.asarray(aff, dtype=np.float64).copy()
    R = aff[:3, :3]
    norms = np.linalg.norm(R, axis=0)
    try:
        singular = abs(float(np.linalg.det(R))) <= 1e-8
    except np.linalg.LinAlgError:
        singular = True
    if not bool(np.isfinite(aff).all()) or singular:
        sp = np.clip(np.where(norms > 1e-6, norms, 1.0), SP_LO, SP_HI)
        signs = np.ones(3)
        for i in range(3):
            if norms[i] > 1e-6:
                signs[i] = np.sign(R[int(np.argmax(np.abs(R[:, i]))), i]) or 1.0
        new = np.eye(4, dtype=np.float64)
        new[:3, :3] = np.diag(sp * signs)
        new[:3, 3] = np.where(np.isfinite(aff[:3, 3]), aff[:3, 3], 0.0)
        return new.astype(np.float32)
    for i in np.where((norms < SP_LO) | (norms > SP_HI))[0]:
        aff[:3, i] = R[:, i] * (float(np.clip(norms[i], SP_LO, SP_HI)) / norms[i])
    return aff.astype(np.float32)


def load_hu_affine(item: dict):
    """item -> (hu float32 (X,Y,Z), affine 4x4). Raw scans are already in HU.
    Extra non-spatial dims (a 4D NIfTI — singleton frame axis or multi-frame series)
    are collapsed to the first frame so MONAI/CT-CLIP see a 3D volume, and degenerate
    affines are sanitized; both occur in the PMBB contrast-abd pool and otherwise kill
    the whole DataLoader worker."""
    if item["kind"] == "nii":
        img = nib.load(item["path"])
        hu = img.get_fdata().astype(np.float32)
        while hu.ndim > 3:
            hu = hu[..., 0]
        return np.ascontiguousarray(hu), _sanitize_affine(img.affine)
    # RAD-ChestCT .npz: 'ct' is (z,y,x) LPS @0.8mm iso -> (x,y,z) + LPS affine
    ct = np.load(item["path"], allow_pickle=True)["ct"].astype(np.float32)
    hu = np.ascontiguousarray(np.transpose(ct, (2, 1, 0)))          # (z,y,x)->(x,y,z)
    s = RADCHEST_SPACING
    aff = np.diag([-s, -s, s, 1.0]).astype(np.float32)              # LPS axial (== CT-RATE)
    return hu, aff


class NativeVolumes(torch.utils.data.Dataset):
    """Per-index native loader; ``transform(item) -> model input tensor`` owns the
    file read so DataLoader workers parallelize the heavy load/resample."""

    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.transform(self.items[i]), i

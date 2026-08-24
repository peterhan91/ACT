"""
Dataset loaders + label/volume alignment for the CT splits.

The h5 files were built in `train_metadata.csv` row order. Failed preprocessing
runs left some h5 files a few rows shorter than the metadata CSV (e.g. CT-RATE
train: 47146 vs 47149; INSPECT test: 3214 vs 3215). The labels CSV has the
same row count as the metadata CSV. So the canonical alignment is:

    1. Read metadata CSV → ordered list of VolumeNames.
    2. Truncate to first `len(h5)` entries.
    3. Look up labels CSV by VolumeName (NOT row position; train_predicted_labels
       can be in a different order).

The result is a dataframe with columns `[VolumeName, h5_idx, <18 labels>]`.
"""
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch

from .paths import (
    DATASETS, INSPECT_OFFICIAL_LABELS_TSV, LABELS_18,
    LABELS_INSPECT_PROGNOSIS, LABELS_PE, LABELS_PE_VERBOSE_MAP,
    cache_volume_index,
)


def load_split(name: str) -> pd.DataFrame:
    """Return aligned (VolumeName, h5_idx, labels) for a split. Cached on disk.

    Dispatches by the dataset's `kind`:
      - default (CT-RATE / INSPECT): VolumeName-based alignment via metadata CSV.
      - "trauma" (RSNA-2023 / RSNA-STR PE): row-order alignment with a
        self-contained labels CSV.
      - "nsclc" (LUNG1 / RADIO): paths CSV is row-aligned with h5; labels
        come from per-labelset split CSVs via load_nsclc_split.
    """
    cfg = DATASETS[name]
    if cfg.get("kind") == "trauma":
        return load_trauma_split(name)
    if cfg.get("kind") == "nsclc":
        return load_nsclc_paths(name)
    if cfg.get("kind") == "radchest":
        return load_radchest_split(name)
    if cfg.get("kind") == "pmbb_ret":
        return load_pmbb_ret(name)

    cache = cache_volume_index(name)
    if cache.exists():
        return pd.read_csv(cache)

    meta = pd.read_csv(cfg["metadata_csv"])
    labels = pd.read_csv(cfg["labels_csv"])

    with h5py.File(cfg["h5"], "r") as f:
        n_h5 = f["ct_volumes"].shape[0]
    if len(meta) < n_h5:
        raise ValueError(
            f"{name}: metadata ({len(meta)}) shorter than h5 ({n_h5}); cannot align."
        )

    meta = meta.head(n_h5).copy()
    meta["h5_idx"] = np.arange(n_h5)

    # INSPECT metadata uses 'image_id' (no .nii.gz suffix) instead of
    # 'VolumeName'. Its labels CSV uses VolumeName=PE<sha>.nii.gz and is NOT
    # in the same row-order as metadata (it has duplicates and a different
    # ordering). Always normalize to a canonical VolumeName then merge.
    if "VolumeName" not in meta.columns:
        meta = meta.copy()
        meta["VolumeName"] = meta["image_id"].astype(str) + ".nii.gz"
    df = meta[["VolumeName", "h5_idx"]].merge(
        labels.drop_duplicates("VolumeName")[["VolumeName"] + LABELS_18],
        on="VolumeName", how="left",
    )
    missing = df[LABELS_18].isna().any(axis=1).sum()
    if missing:
        print(f"[data.load_split:{name}] {missing}/{len(df)} rows missing labels; dropped")
        df = df.dropna(subset=LABELS_18).reset_index(drop=True)

    df.to_csv(cache, index=False)
    return df


def load_trauma_split(name: str) -> pd.DataFrame:
    """Loader for RSNA-2023 trauma splits.

    The labels CSV produced by `preprocess_new/generate_label_csvs.py` is
    row-for-row aligned with the paths CSV used to build the h5 (volumes were
    written into h5 in the same order). So h5_idx == csv row index, and we
    don't need any VolumeName remapping. If the h5 ended up shorter than the
    labels CSV (some volumes failed preprocessing), we truncate to the shorter
    of the two.

    Returns a frame with `[Path, patient_id, series_id, h5_idx, <label cols>]`.
    """
    cfg = DATASETS[name]
    labels = pd.read_csv(cfg["labels_csv"])
    with h5py.File(cfg["h5"], "r") as f:
        n_h5 = f["ct_volumes"].shape[0]
    if len(labels) != n_h5:
        n = min(len(labels), n_h5)
        print(f"[data.load_trauma_split:{name}] h5 ({n_h5}) vs labels "
              f"({len(labels)}) mismatch — truncating to {n}")
        labels = labels.head(n).copy()
    labels = labels.reset_index(drop=True)
    labels["h5_idx"] = np.arange(len(labels), dtype=np.int64)
    return labels


def load_radchest_split(name: str) -> pd.DataFrame:
    """Loader for RAD-ChestCT external validation (same 18 CT-RATE labels).

    The h5 carries a `filenames` dataset (e.g. 'trn00022.npz') identifying each
    volume; the labels CSV is keyed by `NoteAcc_DEID` (e.g. 'trn00022') and is
    NOT in the same row order as the h5. We therefore align by ID (strip the
    '.npz' suffix) — the same VolumeName-style join used for CT-RATE/INSPECT,
    NOT the row-order alignment used for trauma splits. `h5_idx` follows the
    h5's own ordering so it indexes `ct_volumes` directly.

    Returns `[NoteAcc_DEID, h5_idx, <18 labels>]`.
    """
    cfg = DATASETS[name]
    with h5py.File(cfg["h5"], "r") as f:
        n_h5 = f["ct_volumes"].shape[0]
        fns = f["filenames"][:]
    ids = [(x.decode() if isinstance(x, bytes) else str(x)) for x in fns]
    ids = [i[:-4] if i.endswith(".npz") else i for i in ids]
    base = pd.DataFrame({"NoteAcc_DEID": ids,
                         "h5_idx": np.arange(n_h5, dtype=np.int64)})

    labels = pd.read_csv(cfg["labels_csv"])
    labels["NoteAcc_DEID"] = labels["NoteAcc_DEID"].astype(str)
    df = base.merge(
        labels.drop_duplicates("NoteAcc_DEID")[["NoteAcc_DEID"] + LABELS_18],
        on="NoteAcc_DEID", how="left",
    )
    missing = df[LABELS_18].isna().any(axis=1).sum()
    if missing:
        print(f"[data.load_radchest_split:{name}] {missing}/{len(df)} rows "
              f"missing labels; dropped")
        df = df.dropna(subset=LABELS_18).reset_index(drop=True)
    for c in LABELS_18:
        df[c] = df[c].astype(np.int64)
    return df


def load_nsclc_paths(name: str) -> pd.DataFrame:
    """Per-patient frame for an NSCLC dataset (no labels). h5_idx == row index.

    Reads <dataset>_paths.csv (one row per patient, written by
    preprocess_new/nsclc_pick_and_convert.py), truncates to h5 size if some
    volumes failed preprocessing.
    """
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["paths_csv"]).reset_index(drop=True)
    with h5py.File(cfg["h5"], "r") as f:
        n_h5 = f["ct_volumes"].shape[0]
    if len(df) != n_h5:
        n = min(len(df), n_h5)
        print(f"[load_nsclc_paths:{name}] h5 ({n_h5}) vs paths ({len(df)}) — truncating to {n}")
        df = df.head(n).copy()
    df["h5_idx"] = np.arange(len(df), dtype=np.int64)
    return df


def load_pmbb_ret(name: str) -> pd.DataFrame:
    """PMBB non-contrast retrieval pool (exp2). The manifest already carries the
    original `h5_idx` into pmbb_ct_volumes_iso_spacing.h5 plus the paired
    impression, so this is a pass-through. Returns
    `[VolumeName, h5_idx, Path, region, patient, study, Impressions_EN]`.
    Asserts every h5_idx is in range of the h5 (no truncation/alignment guess)."""
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["manifest_csv"])
    df["h5_idx"] = df["h5_idx"].astype(np.int64)
    with h5py.File(cfg["h5"], "r") as f:
        n_h5 = f["ct_volumes"].shape[0]
    bad = int((df["h5_idx"] < 0).sum() + (df["h5_idx"] >= n_h5).sum())
    if bad:
        raise ValueError(f"{name}: {bad} manifest h5_idx out of range [0,{n_h5})")
    return df.reset_index(drop=True)


def load_nsclc_split(dataset: str, labelset: str, split: str) -> pd.DataFrame:
    """Per-labelset 80/20 split frame for NSCLC, joined to h5 indices.

    Returns [PatientID, h5_idx, Path, <label cols>] for the requested split.
    """
    cfg = DATASETS[dataset]
    paths = load_nsclc_paths(dataset)[["PatientID", "h5_idx", "Path"]]
    split_csv = Path(cfg["splits_dir"]) / f"{dataset}_{labelset}_split.csv"
    splits = pd.read_csv(split_csv)
    splits = splits[splits["split"] == split]
    df = splits.merge(paths, on="PatientID", how="inner")
    missing = len(splits) - len(df)
    if missing:
        print(f"[load_nsclc_split:{dataset}/{labelset}/{split}] "
              f"dropping {missing} rows w/o h5")
    return df.reset_index(drop=True)


def load_pe_split(name: str) -> pd.DataFrame:
    """Like load_split but joins INSPECT's PE-task labels (3 binary columns)
    instead of the 18 RadBERT-predicted CT-RATE pathology labels.

    Returns a frame with `[VolumeName, h5_idx, pe_positive, pe_acute,
    pe_subsegmentalonly]`. Validation file uses verbose label names; we
    canonicalize via LABELS_PE_VERBOSE_MAP.
    """
    cfg = DATASETS[name]
    if "pe_labels_csv" not in cfg:
        raise ValueError(f"{name}: no pe_labels_csv in config")
    pe = pd.read_csv(cfg["pe_labels_csv"])
    pe = pe.rename(columns=LABELS_PE_VERBOSE_MAP)
    pe = pe[["VolumeName"] + LABELS_PE].drop_duplicates("VolumeName")

    base = load_split(name)[["VolumeName", "h5_idx"]]
    df = base.merge(pe, on="VolumeName", how="left")
    miss = df[LABELS_PE].isna().any(axis=1).sum()
    if miss:
        print(f"[data.load_pe_split:{name}] dropping {miss} rows w/ missing PE labels")
        df = df.dropna(subset=LABELS_PE).reset_index(drop=True)
    return df


def load_prognosis_split(name: str) -> pd.DataFrame:
    """Join the official INSPECT prognosis labels onto an inspect_* split.

    The official `labels_20250611.tsv` is keyed by impression_id; our inspect
    metadata CSVs already carry impression_id, so we look up VolumeName →
    impression_id → 7 binary prognosis flags. Per-label NaN is retained (some
    flags are censored for some patients), so downstream code can drop NaN
    rows per-task.

    Returns `[VolumeName, h5_idx, impression_id, <7 prognosis cols>]`.
    """
    cfg = DATASETS[name]
    meta_cols = pd.read_csv(cfg["metadata_csv"], nrows=1).columns
    if "impression_id" not in meta_cols:
        raise ValueError(f"{name}: metadata has no impression_id (not INSPECT?)")
    base = load_split(name)[["VolumeName", "h5_idx"]]
    meta = pd.read_csv(cfg["metadata_csv"], usecols=lambda c: c in {
        "VolumeName", "image_id", "impression_id",
    })
    if "VolumeName" not in meta.columns:
        meta = meta.copy()
        meta["VolumeName"] = meta["image_id"].astype(str) + ".nii.gz"
    base = base.merge(
        meta[["VolumeName", "impression_id"]].drop_duplicates("VolumeName"),
        on="VolumeName", how="left",
    )
    labels = pd.read_csv(INSPECT_OFFICIAL_LABELS_TSV, sep="\t",
                         usecols=["impression_id"] + LABELS_INSPECT_PROGNOSIS)
    # Cast TRUE/FALSE/Censored → 1/0/NaN so per-label AUC can drop censored rows.
    _MAP = {"TRUE": 1.0, "FALSE": 0.0, "Censored": np.nan}
    for c in LABELS_INSPECT_PROGNOSIS:
        labels[c] = labels[c].map(_MAP).astype("float32")
    df = base.merge(labels.drop_duplicates("impression_id"),
                    on="impression_id", how="left")
    miss = df[LABELS_INSPECT_PROGNOSIS].isna().all(axis=1).sum()
    if miss:
        print(f"[data.load_prognosis_split:{name}] dropping {miss} rows w/o "
              f"prognosis match")
        df = df.dropna(subset=LABELS_INSPECT_PROGNOSIS, how="all").reset_index(drop=True)
    return df


class CTH5Dataset(torch.utils.data.Dataset):
    """Lazy h5 reader; opens file once per worker. Mirrors eval_all.py's loader."""

    def __init__(self, h5_path: str, h5_indices: np.ndarray):
        self.h5_path = h5_path
        self.h5_indices = np.asarray(h5_indices, dtype=np.int64)
        self._h5 = None

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")["ct_volumes"]

    def __len__(self):
        return len(self.h5_indices)

    def __getitem__(self, idx):
        self._ensure_open()
        h5_idx = int(self.h5_indices[idx])
        img = self._h5[h5_idx]                 # (160, 224, 224) uint8
        img = np.expand_dims(img, 0)           # (1, 160, 224, 224)
        img = np.repeat(img, 3, axis=0)        # (3, 160, 224, 224)
        return {"img": torch.from_numpy(img).float(), "idx": idx}

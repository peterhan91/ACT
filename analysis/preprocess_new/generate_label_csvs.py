#!/usr/bin/env python3
"""Generate per-split label CSVs aligned (row-for-row) with the path CSVs.

For each split, writes <dataset>_<split>_labels.csv whose i-th row gives the
label(s) for the i-th volume in <dataset>_<split>_paths.csv. Downstream
zero-shot / linear-probing scripts can then index labels by h5 row index.

Datasets:
  --dataset multimodalpe  -- Stanford PE: idx, label (PE present 0/1), pe_type
  --dataset rsna2023      -- RSNA-2023 abdominal trauma: bowel/extravasation/
                              kidney/liver/spleen multi-task labels + any_injury
  --dataset coca          -- COCA gated: cac_present (binary, XML existence) +
                              total_agatston (sum over ROIs of weight x area)
                              Nongated patients have no labels available, so
                              their rows get all-NaN entries.
"""
from __future__ import annotations

import argparse
import plistlib
from pathlib import Path

import numpy as np
import pandas as pd


# ---- MultimodalPE ----------------------------------------------------------

def labels_multimodalpe(paths_csv: str, labels_csv: str) -> pd.DataFrame:
    paths = pd.read_csv(paths_csv)
    if paths.empty:
        raise SystemExit(
            f"[labels_multimodalpe] {paths_csv} is empty — upstream NPY->NIfTI step "
            f"produced 0 NIfTI files. Inspect the conversion log before continuing."
        )
    paths["fname"] = paths["Path"].apply(lambda p: Path(p).name.replace(".nii.gz", ""))
    paths["idx"] = paths["fname"].apply(lambda n: int(n.split("__")[-1]))
    labels = pd.read_csv(labels_csv)[["idx", "label", "pe_type"]]
    merged = paths.merge(labels, on="idx", how="left")
    return merged[["Path", "idx", "label", "pe_type"]]


# ---- RSNA2023 --------------------------------------------------------------

_RSNA_LABEL_COLS = [
    "bowel_healthy", "bowel_injury",
    "extravasation_healthy", "extravasation_injury",
    "kidney_healthy", "kidney_low", "kidney_high",
    "liver_healthy", "liver_low", "liver_high",
    "spleen_healthy", "spleen_low", "spleen_high",
    "any_injury",
]


def labels_rsna2023(paths_csv: str, labels_csv: str) -> pd.DataFrame:
    paths = pd.read_csv(paths_csv)
    if paths.empty:
        raise SystemExit(
            f"[labels_rsna2023] {paths_csv} is empty — upstream DICOM->NIfTI step "
            f"produced 0 NIfTI files. Inspect the conversion log / _failures.txt "
            f"before continuing."
        )
    paths["fname"] = paths["Path"].apply(lambda p: Path(p).name.replace(".nii.gz", ""))
    parts = paths["fname"].str.split("__", expand=True)
    paths["patient_id"] = parts[0].astype(int)
    paths["series_id"] = parts[1].astype(int) if parts.shape[1] > 1 else -1
    labels = pd.read_csv(labels_csv)
    cols = ["patient_id"] + _RSNA_LABEL_COLS
    merged = paths.merge(labels[cols], on="patient_id", how="left")
    return merged[["Path", "patient_id", "series_id"] + _RSNA_LABEL_COLS]


# ---- COCA ------------------------------------------------------------------

# Agatston density-based weighting on per-lesion peak HU.
def _agatston_weight(max_hu: float) -> int:
    if max_hu < 130:
        return 0
    if max_hu < 200:
        return 1
    if max_hu < 300:
        return 2
    if max_hu < 400:
        return 3
    return 4


def _parse_coca_xml(xml_path: Path) -> dict:
    """Return {n_rois, total_agatston (raw, area unit = whatever XML uses),
    per_vessel_score: {vessel_name: score}}."""
    with open(xml_path, "rb") as f:
        try:
            data = plistlib.load(f)
        except Exception:
            return {"n_rois": 0, "total_agatston": np.nan, "per_vessel": {}}
    images = data.get("Images", []) if isinstance(data, dict) else []
    total = 0.0
    n = 0
    per_vessel: dict[str, float] = {}
    for img in images:
        for roi in img.get("ROIs", []):
            try:
                area = float(roi.get("Area", 0.0))
                max_hu = float(roi.get("Max", 0.0))
                name = str(roi.get("Name", "Unknown"))
            except Exception:
                continue
            w = _agatston_weight(max_hu)
            score = w * area
            total += score
            per_vessel[name] = per_vessel.get(name, 0.0) + score
            n += 1
    return {"n_rois": n, "total_agatston": total, "per_vessel": per_vessel}


def labels_coca(paths_csv: str, gated_xml_dir: str) -> pd.DataFrame:
    """Build labels for the unified coca paths CSV (gated + nongated)."""
    paths = pd.read_csv(paths_csv)
    if paths.empty:
        raise SystemExit(
            f"[labels_coca] {paths_csv} is empty — upstream DICOM->NIfTI step "
            f"produced 0 NIfTI files. Inspect the conversion log / _failures.txt "
            f"before continuing."
        )

    # Filename layouts (from dicom_series_to_nifti.py):
    #   gated   : Gated_release_final__patient__<pid>__<series>.nii.gz
    #   nongated: deidentified_nongated__<pid>...nii.gz
    def parse(p: str) -> tuple[str, str]:
        name = Path(p).name.replace(".nii.gz", "")
        parts = name.split("__")
        if parts[0].startswith("Gated_release_final") and len(parts) >= 3:
            return ("gated", parts[2])
        if parts[0].startswith("deidentified_nongated") and len(parts) >= 2:
            return ("nongated", parts[1])
        return ("other", parts[0])

    parsed = paths["Path"].apply(parse)
    paths["subset"] = [t[0] for t in parsed]
    paths["patient_id"] = [t[1] for t in parsed]

    # Vessel score columns we care about; coca XML names them in full.
    vessel_keys = ["Right Coronary Artery", "Left Anterior Descending Artery",
                   "Left Circumflex Artery", "Left Main Artery"]
    short_names = {
        "Right Coronary Artery": "rca",
        "Left Anterior Descending Artery": "lad",
        "Left Circumflex Artery": "lcx",
        "Left Main Artery": "lm",
    }

    rows = []
    xml_root = Path(gated_xml_dir)
    for _, r in paths.iterrows():
        rec = {"Path": r["Path"], "patient_id": r["patient_id"], "subset": r["subset"]}
        if r["subset"] == "gated":
            xml_path = xml_root / f"{r['patient_id']}.xml"
            if xml_path.exists():
                parsed = _parse_coca_xml(xml_path)
                rec["cac_present"] = 1 if parsed["n_rois"] > 0 else 0
                rec["n_rois"] = parsed["n_rois"]
                rec["total_agatston"] = parsed["total_agatston"]
                for key in vessel_keys:
                    rec[short_names[key]] = parsed["per_vessel"].get(key, 0.0)
            else:
                # No XML -> assumed CAC=0 (Stanford convention: only patients
                # with calcium have an XML produced).
                rec["cac_present"] = 0
                rec["n_rois"] = 0
                rec["total_agatston"] = 0.0
                for key in vessel_keys:
                    rec[short_names[key]] = 0.0
        else:
            # Nongated: no labels available in this dataset bundle.
            rec["cac_present"] = np.nan
            rec["n_rois"] = np.nan
            rec["total_agatston"] = np.nan
            for key in vessel_keys:
                rec[short_names[key]] = np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    return df[["Path", "patient_id", "subset", "cac_present", "n_rois",
               "total_agatston", "rca", "lad", "lcx", "lm"]]


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["multimodalpe", "rsna2023", "coca"])
    ap.add_argument("--paths_csv", required=True,
                    help="One <dataset>_<split>_paths.csv at a time")
    ap.add_argument("--out_csv", required=True,
                    help="Destination labels CSV (row-for-row alignment with paths_csv)")
    ap.add_argument("--source_labels_csv", default=None,
                    help="multimodalpe: Labels.csv  |  rsna2023: train_2024.csv")
    ap.add_argument("--coca_xml_dir", default=None,
                    help="coca: path to Gated_release_final/calcium_xml")
    args = ap.parse_args()

    if args.dataset == "multimodalpe":
        if not args.source_labels_csv:
            raise SystemExit("--source_labels_csv (Labels.csv) required for multimodalpe")
        df = labels_multimodalpe(args.paths_csv, args.source_labels_csv)
    elif args.dataset == "rsna2023":
        if not args.source_labels_csv:
            raise SystemExit("--source_labels_csv (train_2024.csv) required for rsna2023")
        df = labels_rsna2023(args.paths_csv, args.source_labels_csv)
    elif args.dataset == "coca":
        if not args.coca_xml_dir:
            raise SystemExit("--coca_xml_dir required for coca")
        df = labels_coca(args.paths_csv, args.coca_xml_dir)
    else:
        raise SystemExit(f"unknown dataset {args.dataset}")

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    n_with_label = df.iloc[:, 1:].notna().any(axis=1).sum()
    print(f"[write] {args.out_csv}  rows={len(df)}  with_any_label={n_with_label}")


if __name__ == "__main__":
    main()

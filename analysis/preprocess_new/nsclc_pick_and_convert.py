#!/usr/bin/env python3
"""For NSCLC TCIA datasets (LUNG1, RADIO): walk DICOMs, pick one canonical CT
series per patient, convert it to NIfTI with SimpleITK, emit a paths CSV that
is row-aligned with the NIfTI output order for run_preprocess.py.

Canonical-CT heuristic per patient:
    Modality == 'CT'
    Rows == Cols == 512          # excludes coronal reformats + PET
    n_slices >= 50               # excludes localizers/scouts
    SeriesDescription has none of {FUSION, PET, MAC, NAC, ATTEN, WB}
Tiebreak: thinnest SliceThickness, then largest n_slices, then UID.
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pydicom
import SimpleITK as sitk

EXCLUDE_DESC = re.compile(r"(FUSION|PET|MAC|NAC|ATTEN|WB)", re.IGNORECASE)


def enumerate_patient_series(patient_dir: Path) -> list[dict]:
    series: dict[Path, list[Path]] = {}
    for dcm in patient_dir.rglob("*.dcm"):
        series.setdefault(dcm.parent, []).append(dcm)
    out = []
    for sdir, files in series.items():
        if len(files) < 2:           # RTSTRUCT / SEG / scout-single
            continue
        try:
            ds = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
        except Exception:
            continue
        out.append(dict(
            series_dir=sdir,
            n=len(files),
            modality=str(getattr(ds, "Modality", "")),
            rows=int(getattr(ds, "Rows", 0) or 0),
            cols=int(getattr(ds, "Columns", 0) or 0),
            desc=str(getattr(ds, "SeriesDescription", "") or ""),
            thickness=float(getattr(ds, "SliceThickness", 0) or 0),
            suid=str(getattr(ds, "SeriesInstanceUID", "")),
        ))
    return out


def pick_canonical(series: list[dict]) -> dict | None:
    cand = [s for s in series
            if s["modality"] == "CT"
            and s["rows"] == 512 and s["cols"] == 512
            and s["n"] >= 50
            and not EXCLUDE_DESC.search(s["desc"])]
    if not cand:
        return None
    cand.sort(key=lambda s: (s["thickness"] if s["thickness"] > 0 else 9999,
                              -s["n"], s["suid"]))
    return cand[0]


def convert_one(args) -> tuple[str, str]:
    series_dir, out_nii = Path(args[0]), Path(args[1])
    if out_nii.exists():
        return out_nii.name, "skip"
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetImageIO("GDCMImageIO")
        ids = reader.GetGDCMSeriesIDs(str(series_dir))
        if not ids:
            return out_nii.name, "no series id"
        files = reader.GetGDCMSeriesFileNames(str(series_dir), ids[0])
        reader.SetFileNames(files)
        img = reader.Execute()
        sitk.WriteImage(img, str(out_nii), useCompression=True)
        return out_nii.name, "ok"
    except Exception as e:
        return out_nii.name, f"err: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Root containing one subdir per patient (LUNG1-001/, AMC-001/, ...)")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write <PatientID>.nii.gz")
    ap.add_argument("--paths_csv", required=True,
                    help="Output CSV: PatientID,Path — row-aligned with NIfTI output order")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    in_root = Path(args.input_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    Path(args.paths_csv).parent.mkdir(parents=True, exist_ok=True)

    patients = sorted(p for p in in_root.iterdir() if p.is_dir())
    print(f"[pick] {len(patients)} patient dirs under {in_root}")

    picks = []
    for pt in patients:
        chosen = pick_canonical(enumerate_patient_series(pt))
        if chosen is None:
            print(f"  SKIP {pt.name}")
            continue
        picks.append((pt.name, chosen["series_dir"], out_root / f"{pt.name}.nii.gz"))
    print(f"[pick] selected {len(picks)} / {len(patients)} patients")

    jobs = [(s, n) for _, s, n in picks]
    ok = err = skip = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for fut in as_completed(ex.submit(convert_one, j) for j in jobs):
            name, status = fut.result()
            if status == "ok":   ok += 1
            elif status == "skip": skip += 1
            else:                err += 1; print(f"  {name}: {status}")
    print(f"[convert] ok={ok} skip={skip} err={err}")

    rows = [(pid, str(nii)) for pid, _, nii in picks if nii.exists()]
    pd.DataFrame(rows, columns=["PatientID", "Path"]).to_csv(args.paths_csv, index=False)
    print(f"[write] {len(rows)} rows → {args.paths_csv}")


if __name__ == "__main__":
    main()

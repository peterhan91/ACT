#!/usr/bin/env python3
"""Universal DICOM-series → NIfTI batch converter (SimpleITK + pydicom engine).

Walks `--input_dir` and finds every "leaf directory" containing >= --min_slices
DICOM files (`*.dcm`). Converts each leaf directory into one .nii.gz file
under `--output_dir`, named by the sanitized relative path from `--input_dir`.

Why SimpleITK rather than dicom2nifti:
  - RSNA-2023 DICOMs are heavily anonymized (no Modality tag) → dicom2nifti
    rejects them. SimpleITK's GDCM ImageIO accepts them.
  - COCA series occasionally fail dicom2nifti's slice-uniformity check.

Pipeline per series:
  1. List `*.dcm` files.
  2. Sort by ImagePositionPatient[2] (fallback: InstanceNumber, then filename).
  3. Build a SimpleITK Image via ImageSeriesReader (GDCM ImageIO automatically
     applies DICOM RescaleSlope/Intercept → output is in HU).
  4. If slice non-uniformity is large, override z-spacing with the median
     adjacent-slice IPP delta (more reliable on RSNA's noisy headers).
  5. Save as .nii.gz. Spacing/affine in the NIfTI header round-trip via
     downstream `nib.load` + `nib.as_closest_canonical` (run_preprocess.py).

Output filename = sanitized relative path from input_dir, e.g.
    .../train_images/10004/21057/  →  10004__21057.nii.gz
Also emits `<output_dir>/_paths.csv` with a `Path` column ready to feed
run_preprocess.py via `--ct_data_path <paths_csv>`.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import statistics
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# --- ensure SimpleITK + pydicom available -----------------------------------
def _ensure_deps():
    """Fail fast if SimpleITK or pydicom isn't importable in the active env.

    The previous behavior (`pip install --user --quiet ...`) was fragile under
    slurm: `--user` resolves to a different filesystem path on login vs.
    compute nodes, `--quiet` masks pip warnings, and a partial install (one
    package landed, the other didn't) silently propagated, leading to 100%
    worker failures with a confusing ModuleNotFoundError per series. Install
    these once into the env (pip install SimpleITK>=2.3 pydicom>=2.4, *without*
    --user) and we'll just verify here.
    """
    missing = [m for m in ("SimpleITK", "pydicom") if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(
            f"[deps] {missing} not importable in {sys.executable} (host {os.uname().nodename}).\n"
            f"       Install into the env directly:\n"
            f"           conda activate ctproject\n"
            f"           python -m pip install {' '.join(missing)}\n"
            f"       Do NOT rely on `pip install --user` — its target dir varies between login\n"
            f"       and compute nodes on this cluster."
        )


def _is_leaf_dicom_dir(path: Path, min_slices: int) -> bool:
    """A 'leaf series dir' has >= min_slices *.dcm and no DICOM-bearing subdirs."""
    if not path.is_dir():
        return False
    try:
        entries = list(path.iterdir())
    except (PermissionError, OSError):
        return False
    n_dcm = sum(1 for e in entries if e.is_file() and e.suffix.lower() == ".dcm")
    if n_dcm < min_slices:
        return False
    for e in entries:
        if e.is_dir():
            try:
                if any(c.suffix.lower() == ".dcm" for c in e.iterdir() if c.is_file()):
                    return False
            except (PermissionError, OSError):
                continue
    return True


def _find_series_dirs(root: Path, min_slices: int) -> list[Path]:
    out: list[Path] = []
    for cur, _dirs, files in os.walk(root):
        cur_p = Path(cur)
        n_dcm = sum(1 for f in files if f.lower().endswith(".dcm"))
        if n_dcm >= min_slices and _is_leaf_dicom_dir(cur_p, min_slices):
            out.append(cur_p)
    out.sort()
    return out


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(rel_path: Path) -> str:
    s = str(rel_path).strip("/").replace("/", "__")
    return _SAFE.sub("_", s)


def _sort_dicom_files(dcm_files: list[str]) -> tuple[list[str], list[float] | None]:
    """Sort DICOMs by ImagePositionPatient[2], InstanceNumber, or filename.

    Returns (sorted_filenames, ipp_z_list_or_None). ipp_z_list is the list of
    sorted z-coordinates (only set if every file had a parseable IPP).
    """
    import pydicom
    rows: list[tuple[float | None, int | None, str, str]] = []
    all_have_ipp = True
    for fn in dcm_files:
        try:
            ds = pydicom.dcmread(fn, stop_before_pixels=True, force=True)
        except Exception:
            rows.append((None, None, fn, fn))
            all_have_ipp = False
            continue
        ipp = getattr(ds, "ImagePositionPatient", None)
        z = float(ipp[2]) if ipp is not None and len(ipp) == 3 else None
        if z is None:
            all_have_ipp = False
        inst = getattr(ds, "InstanceNumber", None)
        try:
            inst = int(inst) if inst is not None else None
        except Exception:
            inst = None
        rows.append((z, inst, fn, fn))

    if all_have_ipp:
        rows.sort(key=lambda r: r[0])  # by z
        return [r[2] for r in rows], [r[0] for r in rows]
    # fall back to InstanceNumber, then filename
    rows.sort(key=lambda r: (r[1] if r[1] is not None else 1 << 30, r[3]))
    return [r[2] for r in rows], None


def _convert_one(args):
    series_dir, out_path = args
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        return (str(series_dir), str(out_path), "SKIP_EXISTS", None)
    try:
        import SimpleITK as sitk
        # 1) collect DICOMs
        dcm_files = sorted(
            os.path.join(str(series_dir), f)
            for f in os.listdir(str(series_dir))
            if f.lower().endswith(".dcm")
        )
        if len(dcm_files) < 2:
            return (str(series_dir), str(out_path), "FAIL", "fewer than 2 DICOMs")

        # 2) sort by geometry
        sorted_files, ipp_z = _sort_dicom_files(dcm_files)

        # 3) read as a series (GDCM applies RescaleSlope/Intercept → HU)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sorted_files)
        reader.MetaDataDictionaryArrayUpdateOn()
        img = reader.Execute()
        sz = img.GetSize()
        sp = list(img.GetSpacing())  # (x, y, z)

        # 4) if slice spacing looks bogus (e.g. RSNA with 0.1 mm), override
        #    with the median |Δz| from sorted IPP
        if ipp_z is not None and len(ipp_z) >= 2:
            deltas = [abs(ipp_z[i + 1] - ipp_z[i]) for i in range(len(ipp_z) - 1) if abs(ipp_z[i + 1] - ipp_z[i]) > 1e-6]
            if deltas:
                med_dz = float(statistics.median(deltas))
                # only override if SITK's z-spacing is implausibly small (< 0.5 mm)
                # or differs from median by > 50%
                if sp[2] < 0.5 or abs(sp[2] - med_dz) / max(med_dz, 1e-6) > 0.5:
                    sp[2] = med_dz
                    img.SetSpacing(tuple(sp))

        # 5) write NIfTI
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(out_path), useCompression=True)
        ok = out_path.exists() and out_path.stat().st_size > 0
        return (str(series_dir), str(out_path),
                "OK" if ok else "EMPTY",
                f"shape={sz} spacing={tuple(round(s,4) for s in sp)} n_dcm={len(sorted_files)}")
    except Exception as e:  # noqa: BLE001
        return (str(series_dir), str(out_path), "FAIL",
                f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--paths_csv", required=True,
                    help="Output CSV with one `Path` column listing each .nii.gz")
    ap.add_argument("--min_slices", type=int, default=10,
                    help="Skip series with fewer DICOM slices (filters dose reports / scouts)")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: convert only the first N series")
    args = ap.parse_args()

    _ensure_deps()

    in_root = Path(args.input_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    Path(args.paths_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"[scan] discovering DICOM series under {in_root} (min_slices={args.min_slices})", flush=True)
    series_dirs = _find_series_dirs(in_root, args.min_slices)
    if args.limit > 0:
        series_dirs = series_dirs[: args.limit]
    print(f"[scan] {len(series_dirs)} series found", flush=True)

    jobs: list[tuple[Path, Path]] = []
    for sd in series_dirs:
        rel = sd.relative_to(in_root)
        # if input_dir itself is the series dir, rel == Path('.') → use input_dir's name
        name = _safe_name(rel) if str(rel) not in (".", "") else _safe_name(Path(in_root.name))
        out_path = out_root / f"{name}.nii.gz"
        jobs.append((sd, out_path))

    n_ok = n_skip = n_fail = 0
    fail_log = Path(args.output_dir) / "_failures.txt"
    succeeded: list[Path] = []

    print(f"[convert] running with {args.num_workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_convert_one, j): j for j in jobs}
        with open(fail_log, "w") as flog, tqdm(total=len(jobs), desc="series") as bar:
            for fut in as_completed(futures):
                sd, op, status, info = fut.result()
                if status == "OK":
                    n_ok += 1
                    succeeded.append(Path(op))
                elif status == "SKIP_EXISTS":
                    n_skip += 1
                    succeeded.append(Path(op))
                else:
                    n_fail += 1
                    flog.write(f"{sd}\t{status}\t{info or ''}\n")
                    flog.flush()
                bar.update(1)

    print(f"[convert] done. OK={n_ok}  SKIP_EXISTS={n_skip}  FAIL={n_fail}  → see {fail_log}", flush=True)

    # Hard-fail if every series failed — otherwise downstream steps silently
    # produce 0-row split CSVs and the next stage crashes with a confusing
    # `KeyError: 0` / `ValueError: not enough values to unpack`.
    if n_ok == 0 and n_skip == 0 and len(jobs) > 0:
        sys.exit(
            f"[convert] EVERY one of {len(jobs)} series failed conversion — see {fail_log}.\n"
            f"          Most common cause: SimpleITK/pydicom missing in the worker env."
        )

    succeeded.sort()
    pd.DataFrame({"Path": [str(p) for p in succeeded]}).to_csv(args.paths_csv, index=False)
    print(f"[paths] wrote {len(succeeded)} entries to {args.paths_csv}", flush=True)


if __name__ == "__main__":
    main()

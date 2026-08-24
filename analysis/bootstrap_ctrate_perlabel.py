#!/usr/bin/env python3
"""
Patient-clustered bootstrap 95% CIs for CT-RATE per-label linear-probe AUROC.

Unlike the seed-std error bars in plot_probe_perlabel_grid.py (training-init
variance across the 20 fairA reruns), this quantifies PATIENT-SAMPLING variance:
if we drew a different sample of patients from the population, how much would
each label's AUROC move? Same statistical target as exp3's patient-clustered
concept-retrieval CIs.

Why clustered: CT-RATE's 2,489 scored scans come from only 973 unique patients
(outputs/v1/cache/volume_index.ctrate_test.csv, VolumeName prefix valid_<id>_),
2-16 scans each (791 patients have exactly 2). Resampling scans independently
would pseudo-replicate multi-scan patients and understate the CI -- so each
bootstrap draw resamples PATIENTS with replacement, then takes ALL of that
patient's scans.

Predictions: mean probability across the 20 fairA seeds per scan (reduces seed
noise so the CI isolates patient-sampling uncertainty, not training variance --
those are already shown separately via the seed-std bars).

Reads   outputs/v1/cache/volume_index.ctrate_test.csv        (patient IDs, GT)
        outputs/v1/probe_compare/fairA/perseed/ctrate__<model>.npz
Writes  outputs/v1/probe_compare/fairA/ctrate_perlabel_bootstrap_ci.csv

Run:  python bootstrap_ctrate_perlabel.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PERSEED = ROOT / "outputs/v1/probe_compare/fairA/perseed"
VOL_INDEX = ROOT / "outputs/v1/cache/volume_index.ctrate_test.csv"
OUT = ROOT / "outputs/v1/probe_compare/fairA/ctrate_perlabel_bootstrap_ci.csv"

MODELS = ["merlin", "ctclip", "fvlm_concat1024", "ours_raw3dclip"]
N_BOOT = 1000
SEED = 0


def _patient_ids(volume_names: np.ndarray) -> np.ndarray:
    return pd.Series(volume_names).str.extract(r"^(valid_\d+)_")[0].to_numpy()


def load_model(model: str):
    z = np.load(PERSEED / f"ctrate__{model}.npz", allow_pickle=True)
    probs = z["probs"].astype(float).mean(axis=0)   # (N, L) mean over 20 seeds
    y = z["y"].astype(int)                            # (N, L)
    labels = [str(x) for x in z["labels"]]
    h5_idx = z["h5_idx"]
    return probs, y, labels, h5_idx


def main() -> None:
    vidx = pd.read_csv(VOL_INDEX).sort_values("h5_idx").reset_index(drop=True)
    patient = _patient_ids(vidx["VolumeName"].to_numpy())          # (N,) aligned to h5_idx order
    uniq_patients = np.unique(patient)
    # rows (scan indices) belonging to each patient, for cluster resampling
    rows_by_patient = {p: np.where(patient == p)[0] for p in uniq_patients}
    n_patients = len(uniq_patients)
    print(f"{n_patients} unique patients, {len(patient)} scans")

    rng = np.random.default_rng(SEED)
    rows = []
    for model in MODELS:
        probs, y, labels, h5_idx = load_model(model)
        assert len(h5_idx) == len(patient), f"{model}: row-count mismatch vs volume_index"

        boot_auc = np.full((N_BOOT, len(labels)), np.nan)
        for b in range(N_BOOT):
            draw = rng.choice(uniq_patients, size=n_patients, replace=True)
            idx = np.concatenate([rows_by_patient[p] for p in draw])
            yb, pb = y[idx], probs[idx]
            for li in range(len(labels)):
                if 0 < yb[:, li].sum() < len(idx):     # both classes present
                    boot_auc[b, li] = roc_auc_score(yb[:, li], pb[:, li])

        for li, lab in enumerate(labels):
            vals = boot_auc[:, li]
            vals = vals[~np.isnan(vals)]
            point = roc_auc_score(y[:, li], probs[:, li])
            rows.append({
                "model": model, "label": lab, "point_auc": point,
                "boot_mean": vals.mean(), "ci_lo": np.percentile(vals, 2.5),
                "ci_hi": np.percentile(vals, 97.5), "n_boot_valid": len(vals),
            })
        print(f"done: {model}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print(df.pivot(index="label", columns="model", values="point_auc").round(3))


if __name__ == "__main__":
    main()

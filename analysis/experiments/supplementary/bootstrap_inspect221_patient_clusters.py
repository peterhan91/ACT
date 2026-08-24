#!/usr/bin/env python3
"""Patient-cluster bootstrap for the overall ACT vs CT-CLIP comparison.

Reuses the Supp Table S11 protocol (bootstrap_fig6_all86_patient_clusters.py:
seed 20260721, 1,000 shared multinomial patient resamples of the 2,612-scan /
2,223-patient INSPECT test cohort, exact-tie weighted AUROC, percentile CIs)
and applies it to all 221 phenotypes in both regimes:

    linear probe: perseed probs (20, 2612, 221) for f2llm and ctclip;
                  per-resample AUROC is computed per fit then averaged.
    zero-shot:    fixed score matrices (2612, 221) for f2llm and ctclip.

Models are never refit inside resamples. One shared weight matrix pairs the
two models, so delta distributions are paired. Outputs land in
outputs/inspect221_patient_bootstrap/ (npz distributions + summary.json).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_fig6_all86_patient_clusters import (  # noqa: E402
    build_shared_cluster_weights,
    percentile_interval,
    weighted_auc_for_all_resamples,
)

SEED = 20_260_721
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

ASSETS = REPO / "experiments" / "exp1_zeroshot" / "inspect_pheno"
PERSEED = REPO / "outputs" / "v1" / "probe_compare" / "fairA" / "perseed"
PER_CT_MANIFEST = REPO / "phenotype_labels" / "per_ct" / "manifest.parquet"
ZS_F2LLM = REPO / "outputs" / "v1" / "external" / "phenotype__test__f2llm__probs.npy"
ZS_F2LLM_JSON = REPO / "outputs" / "v1" / "external" / "phenotype__test__f2llm__results.json"
ZS_CTCLIP = REPO / "experiments" / "exp1_zeroshot" / "ctclip" / "results" / "inspect_pheno__native__probs.npy"
ZS_CTCLIP_JSON = REPO / "experiments" / "exp1_zeroshot" / "ctclip" / "results" / "inspect_pheno__native__results.json"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "inspect221_patient_bootstrap"

EXPECTED = {
    "lp_f2llm": 0.7086182082417066,
    "lp_ctclip": 0.6618841981145214,
    "zs_f2llm": 0.6507783543391031,
    "zs_ctclip": 0.5722514203771246,
}


def load_person_ids(volume_names: pd.Series) -> np.ndarray:
    per_ct = pd.read_parquet(PER_CT_MANIFEST)
    per_ct = per_ct[(per_ct["split"] == "test") & per_ct["visit_occurrence_id"].notna()]
    mapping = per_ct.set_index("VolumeName")["person_id"]
    assert not mapping.index.duplicated().any()
    person_ids = mapping.reindex(volume_names).to_numpy()
    assert not pd.isna(person_ids).any(), "unmapped test volume"
    assert len(person_ids) == 2612 and len(np.unique(person_ids)) == 2223
    return person_ids


def load_perseed(model: str, y_ref: np.ndarray, h5_ref: np.ndarray, labels_ref: list[str]) -> np.ndarray:
    z = np.load(PERSEED / f"inspect__{model}.npz", allow_pickle=True)
    assert np.array_equal(z["h5_idx"], h5_ref)
    assert np.array_equal(z["y"], y_ref)
    assert [str(x) for x in z["labels"]] == labels_ref
    probs = np.asarray(z["probs"], dtype=np.float64)
    assert probs.shape == (20, 2612, 221)
    ones = np.ones((1, probs.shape[1]), dtype=np.float64)
    point = np.array([
        [weighted_auc_for_all_resamples(y_ref[:, j], probs[r, :, j], ones)[0] for j in range(221)]
        for r in range(20)
    ])
    # per_label_auc is stored as float32, so per-cell agreement is limited to ~1e-7
    assert np.nanmax(np.abs(point - z["per_label_auc"])) < 1e-6
    assert abs(float(point.mean()) - EXPECTED[f"lp_{model}"]) < 1e-9
    return probs


def load_zeroshot(path_probs: Path, path_json: Path, y_ref: np.ndarray, labels_ref: list[str]) -> np.ndarray:
    scores = np.asarray(np.load(path_probs), dtype=np.float64)
    meta = json.loads(path_json.read_text())
    assert [str(x) for x in meta["labels"]] == labels_ref
    assert scores.shape == (2612, 221)
    ones = np.ones((1, scores.shape[0]), dtype=np.float64)
    point = np.array([weighted_auc_for_all_resamples(y_ref[:, j], scores[:, j], ones)[0] for j in range(221)])
    assert abs(float(np.mean(point)) - meta["mean_auc"]) < 1e-9
    return scores


def boot_per_label_lp(probs: np.ndarray, y: np.ndarray, weights: np.ndarray, tag: str) -> np.ndarray:
    out = np.zeros((weights.shape[0], 221), dtype=np.float64)
    start = time.time()
    for j in range(221):
        acc = np.zeros(weights.shape[0], dtype=np.float64)
        for r in range(probs.shape[0]):
            acc += weighted_auc_for_all_resamples(y[:, j], probs[r, :, j], weights)
        out[:, j] = acc / probs.shape[0]
        if (j + 1) % 20 == 0:
            print(f"[{tag}] {j + 1}/221 labels ({time.time() - start:.0f}s)", flush=True)
    assert np.isfinite(out).all()
    return out


def boot_per_label_zs(scores: np.ndarray, y: np.ndarray, weights: np.ndarray, tag: str) -> np.ndarray:
    out = np.zeros((weights.shape[0], 221), dtype=np.float64)
    for j in range(221):
        out[:, j] = weighted_auc_for_all_resamples(y[:, j], scores[:, j], weights)
    assert np.isfinite(out).all()
    print(f"[{tag}] 221/221 labels", flush=True)
    return out


def summarize(dist: np.ndarray, point: float) -> dict:
    low, high, n_finite = percentile_interval(dist, CI_LEVEL)
    return {
        "point": point,
        "boot_mean": float(np.mean(dist)),
        "ci_lower": low,
        "ci_upper": high,
        "n_finite": n_finite,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ASSETS / "manifest.csv")
    y = np.load(ASSETS / "y_221.npy")
    labels = pd.read_csv(ASSETS / "phecodes.csv")["phecode_str"].astype(str).tolist()
    assert y.shape == (2612, 221) and len(labels) == 221
    person_ids = load_person_ids(manifest["VolumeName"])
    h5_ref = manifest["h5_idx"].to_numpy()

    lp = {m: load_perseed(m, y, h5_ref, labels) for m in ("f2llm", "ctclip")}
    zs = {
        "f2llm": load_zeroshot(ZS_F2LLM, ZS_F2LLM_JSON, y, labels),
        "ctclip": load_zeroshot(ZS_CTCLIP, ZS_CTCLIP_JSON, y, labels),
    }
    print("alignment and point-estimate asserts passed", flush=True)

    weights = build_shared_cluster_weights(person_ids, N_BOOTSTRAP, SEED)
    assert weights.shape == (N_BOOTSTRAP, 2612)

    per_label = {}
    for model in ("f2llm", "ctclip"):
        per_label[f"lp_{model}"] = boot_per_label_lp(lp[model], y, weights, f"lp:{model}")
        per_label[f"zs_{model}"] = boot_per_label_zs(zs[model], y, weights, f"zs:{model}")

    summary = {
        "protocol": {
            "seed": SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "ci_level": CI_LEVEL,
            "resampling": "multinomial patient clusters, all scans of a sampled patient share the patient multiplicity",
            "cohort": {"n_test_scans": 2612, "n_test_patients": 2223, "n_phenotypes": 221},
            "linear_probe_estimand": "per-resample AUROC computed per fit for each of the 20 fixed Adam fits, averaged across fits, then averaged across the 221 phenotypes",
            "zero_shot_estimand": "per-resample AUROC of the fixed zero-shot scores, averaged across the 221 phenotypes",
            "models_refit_within_resamples": False,
            "shared_weights_across_models": True,
        },
        "macro": {},
        "paired_delta": {},
        "win_counts": {},
    }
    macro = {k: v.mean(axis=1) for k, v in per_label.items()}
    for key, point in EXPECTED.items():
        summary["macro"][key] = summarize(macro[key], point)
    for regime, point_delta, point_wins in (
        ("lp", EXPECTED["lp_f2llm"] - EXPECTED["lp_ctclip"], 190),
        ("zs", EXPECTED["zs_f2llm"] - EXPECTED["zs_ctclip"], 182),
    ):
        delta = macro[f"{regime}_f2llm"] - macro[f"{regime}_ctclip"]
        summary["paired_delta"][regime] = summarize(delta, point_delta)
        summary["paired_delta"][regime]["excludes_zero"] = bool(
            summary["paired_delta"][regime]["ci_lower"] > 0 or summary["paired_delta"][regime]["ci_upper"] < 0
        )
        wins = (per_label[f"{regime}_f2llm"] > per_label[f"{regime}_ctclip"]).sum(axis=1)
        summary["win_counts"][regime] = summarize(wins.astype(np.float64), float(point_wins))

    np.savez_compressed(
        OUT_DIR / "inspect221_patient_cluster_bootstrap.npz",
        seed=SEED,
        n_bootstrap=N_BOOTSTRAP,
        labels=np.asarray(labels, dtype=object),
        **{f"per_label_{k}": v.astype(np.float32) for k, v in per_label.items()},
        **{f"macro_{k}": v for k, v in macro.items()},
    )
    (OUT_DIR / "inspect221_patient_bootstrap_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["macro"], indent=2))
    print(json.dumps(summary["paired_delta"], indent=2))
    print(json.dumps(summary["win_counts"], indent=2))


if __name__ == "__main__":
    main()

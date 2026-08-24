"""Build per-CT-volume phenotype labels for INSPECT using Merlin's phecode pipeline,
then compare each time-window variant to the collaborator reference labels.

Manifest (VolumeName -> person_id, CT date, visit_occurrence_id) is assembled from:
  /path/to/clip_3d_ct/data/inspect/{train,test,validation}_metadata.csv

We produce three labeling variants and compare each to the collaborator
reference CSV in `/path/to/data/Inspect_v2.0_EHR`:

  A. visit_only      — conditions coded during the same visit_occurrence_id as the CT
  B. prior_all       — every condition on or before the CT date (no future leak)
  C. prior_365d      — conditions within 365 days before CT (plus 7 day grace after)

Outputs (/.../phenotype_labels/per_ct/):
  manifest.parquet                     VolumeName + person_id + CT date + visit id
  per_ct_labels_<variant>.parquet      wide multi-hot: rows = VolumeName, cols = phecode
  compare_<variant>_vs_haifan.csv      phenotype-by-phenotype prevalence compared to the reference
  summary.json                         correlations, MAE, match metrics
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
INS = Path("/path/to/data/Inspect_v2.0_EHR")
LBL = INS / "phenotype_labels"
OUT = LBL / "per_ct"
OUT.mkdir(parents=True, exist_ok=True)
CLIP_DATA = Path("/path/to/clip_3d_ct/data/inspect")
HAIFAN = INS / "inspect_phenotypes_Haifan.csv"


def load_pm():
    spec = importlib.util.spec_from_file_location(
        "pm", Path(__file__).resolve().parent / "phecode_mapping.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    pm = load_pm()
    mapping_path = pm.find_default_phecode_mapping_csv()
    phenotypes_path = pm.find_default_phenotypes_csv()
    curated = pd.read_csv(
        Path(__file__).resolve().parent / "manifests" / "phenotypes.csv", dtype=str
    )
    curated["phecode"] = curated["phecode"].str.strip()
    curated["phecode_str"] = curated["phecode_str"].str.strip()
    curated_set = set(curated["phecode"])
    phecode_to_name = dict(zip(curated["phecode"], curated["phecode_str"]))
    name_to_phecode = dict(zip(curated["phecode_str"], curated["phecode"]))

    icd2phe = pm.load_phecode_mapping(mapping_path)

    # -- 1. build manifest -------------------------------------------------
    print("[1/5] assembling VolumeName manifest...")
    dfs = []
    for name in ("train", "test", "validation"):
        p = CLIP_DATA / f"{name}_metadata.csv"
        d = pd.read_csv(
            p, dtype=str, usecols=["image_id", "person_id_x", "procedure_occurrence_id", "procedure_DATETIME"]
        )
        d["split"] = name
        dfs.append(d)
    man = pd.concat(dfs, ignore_index=True).rename(
        columns={"person_id_x": "person_id", "image_id": "image_id"}
    )
    man["VolumeName"] = man["image_id"] + ".nii.gz"
    man["procedure_DATETIME"] = pd.to_datetime(man["procedure_DATETIME"], errors="coerce")
    # procedure_occurrence_id is a float-string like "124034716.0"; normalize
    man["procedure_occurrence_id"] = (
        man["procedure_occurrence_id"].astype(str).str.rstrip("0").str.rstrip(".")
    )
    # Drop rows missing CT date (can't window without it)
    before = len(man)
    man = man.dropna(subset=["procedure_DATETIME"]).drop_duplicates("VolumeName")
    print(f"       manifest rows: {len(man):,} (dropped {before-len(man)} without CT date)")

    # -- 2. attach visit_occurrence_id via procedure_occurrence ----------
    print("[2/5] attaching visit_occurrence_id from procedure_occurrence.csv...")
    po = pd.read_csv(
        INS / "procedure_occurrence.csv",
        usecols=["procedure_occurrence_id", "visit_occurrence_id"],
        dtype=str,
    )
    po["procedure_occurrence_id"] = po["procedure_occurrence_id"].astype(str).str.rstrip("0").str.rstrip(".")
    man = man.merge(po, on="procedure_occurrence_id", how="left")
    n_visit = man["visit_occurrence_id"].notna().sum()
    print(f"       volumes with visit_occurrence_id: {n_visit:,}/{len(man):,}")
    man.to_parquet(OUT / "manifest.parquet", index=False)

    # -- 3. load all conditions with ICD + date + visit ---------------------
    print("[3/5] loading condition_occurrence with ICD source...")
    concept = pd.read_csv(
        INS / "concept.csv",
        usecols=["concept_id", "vocabulary_id", "concept_code"],
        dtype=str,
    )
    icd = concept[concept["vocabulary_id"].isin(["ICD9CM", "ICD10CM"])]
    co = pd.read_csv(
        INS / "condition_occurrence.csv",
        usecols=[
            "person_id",
            "condition_source_concept_id",
            "condition_start_DATE",
            "visit_occurrence_id",
        ],
        dtype=str,
    )
    co = co.merge(
        icd.rename(columns={"concept_id": "condition_source_concept_id"}),
        on="condition_source_concept_id",
        how="inner",
    )
    co["icd_canon"] = (
        co["concept_code"].str.strip().str.upper().str.replace(".", "", regex=False)
    )
    co["d"] = pd.to_datetime(co["condition_start_DATE"], errors="coerce")
    # Map ICD -> expanded phecodes (restricted to curated set)
    print("       mapping ICD -> curated phecode set (with expansion)...")
    uniq = co["icd_canon"].drop_duplicates().tolist()
    expanded: dict[str, list[str]] = {}
    for code in uniq:
        base = icd2phe.get(code)
        if not base:
            continue
        ex = pm.expand_phecodes(base, phenotypes_path=phenotypes_path)
        ex = [p for p in ex if p in curated_set]
        if ex:
            expanded[code] = ex
    co = co[co["icd_canon"].isin(expanded)].copy()
    co["phecode"] = co["icd_canon"].map(expanded)
    co_exp = co[["person_id", "d", "visit_occurrence_id", "phecode"]].explode(
        "phecode", ignore_index=True
    )
    print(f"       exploded condition-phecode rows: {len(co_exp):,}")

    # -- 4. three variants of per-CT labels -------------------------------
    ordered_phecodes = sorted(
        curated_set,
        key=lambda x: (float(x) if x.replace(".", "", 1).isdigit() else 1e9, x),
    )
    phe_idx = {p: i for i, p in enumerate(ordered_phecodes)}

    def build_variant(name: str, filter_fn):
        print(f"[4/5] variant '{name}': filtering + building wide matrix...")
        # filter_fn takes (co_exp, man_row) and returns True for each co_exp row
        # For efficiency we precompute per variant differently
        raise NotImplementedError

    # --- variant A: visit_only ---
    print("[4a] variant: visit_only")
    visit_join = man.dropna(subset=["visit_occurrence_id"]).merge(
        co_exp, on=["person_id", "visit_occurrence_id"], how="left"
    )
    mat_A = np.zeros((len(man), len(ordered_phecodes)), dtype=np.uint8)
    vn_to_row = {v: i for i, v in enumerate(man["VolumeName"])}
    ok = visit_join.dropna(subset=["phecode"])
    rows = ok["VolumeName"].map(vn_to_row).to_numpy()
    cols = ok["phecode"].map(phe_idx).to_numpy()
    keep = ~pd.isna(cols) & ~pd.isna(rows)
    mat_A[rows[keep].astype(int), cols[keep].astype(int)] = 1

    # --- variant B: prior_all (condition date <= CT date) ---
    print("[4b] variant: prior_all")
    # Build a sorted lookup per person of (date, phecode) and join to man
    # Simpler: join on person_id then filter rows where d <= CT date
    person_to_man = man[["VolumeName", "person_id", "procedure_DATETIME"]]
    joined = co_exp.merge(person_to_man, on="person_id", how="inner")
    mask_B = joined["d"] <= joined["procedure_DATETIME"]
    okB = joined[mask_B].drop_duplicates(["VolumeName", "phecode"])
    mat_B = np.zeros((len(man), len(ordered_phecodes)), dtype=np.uint8)
    rowsB = okB["VolumeName"].map(vn_to_row).to_numpy()
    colsB = okB["phecode"].map(phe_idx).to_numpy()
    keepB = ~pd.isna(colsB) & ~pd.isna(rowsB)
    mat_B[rowsB[keepB].astype(int), colsB[keepB].astype(int)] = 1

    # --- variant C: prior_365d (CT - 365d <= condition date <= CT + 7d) ---
    print("[4c] variant: prior_365d")
    lo = joined["procedure_DATETIME"] - pd.Timedelta(days=365)
    hi = joined["procedure_DATETIME"] + pd.Timedelta(days=7)
    mask_C = (joined["d"] >= lo) & (joined["d"] <= hi)
    okC = joined[mask_C].drop_duplicates(["VolumeName", "phecode"])
    mat_C = np.zeros((len(man), len(ordered_phecodes)), dtype=np.uint8)
    rowsC = okC["VolumeName"].map(vn_to_row).to_numpy()
    colsC = okC["phecode"].map(phe_idx).to_numpy()
    keepC = ~pd.isna(colsC) & ~pd.isna(rowsC)
    mat_C[rowsC[keepC].astype(int), colsC[keepC].astype(int)] = 1

    variants = {"visit_only": mat_A, "prior_all": mat_B, "prior_365d": mat_C}
    for k, mat in variants.items():
        df = pd.DataFrame(
            mat, index=pd.Index(man["VolumeName"].values, name="VolumeName"), columns=ordered_phecodes
        )
        df.to_parquet(OUT / f"per_ct_labels_{k}.parquet")
        print(f"       wrote {OUT / f'per_ct_labels_{k}.parquet'}  "
              f"({df.shape}, total positives={int(df.values.sum()):,})")

    # -- 5. compare each variant to the reference -------------------------
    print("[5/5] comparing to the reference...")
    ref = pd.read_csv(HAIFAN)
    ref_cols = [c for c in ref.columns if c != "VolumeName"]
    overlap = [c for c in ref_cols if c in name_to_phecode]
    # Align on VolumeName
    ref = ref.set_index("VolumeName")
    common_volumes = sorted(set(man["VolumeName"]) & set(ref.index))
    print(f"       common volumes (ours ∩ reference): {len(common_volumes):,}")
    summary: dict = {
        "n_volumes_in_manifest": len(man),
        "n_volumes_in_haifan": len(ref),
        "n_common_volumes": len(common_volumes),
        "variants": {},
    }

    ref_common = ref.loc[common_volumes, overlap].apply(pd.to_numeric, errors="coerce")
    ref_pos = (ref_common > 0).astype(np.uint8).values

    for k, mat in variants.items():
        ours_df = pd.DataFrame(
            mat, index=man["VolumeName"].values, columns=ordered_phecodes
        ).loc[common_volumes]
        # Select the phecodes corresponding to the reference's overlapping column names
        cols_in_order = [name_to_phecode[n] for n in overlap]
        ours_mat = ours_df[cols_in_order].values.astype(np.uint8)

        # Row-level agreement: exact-match rate per volume-per-column
        agree = (ours_mat == ref_pos).mean()
        # Per-column prevalence
        ref_prev = ref_pos.mean(axis=0)
        our_prev = ours_mat.mean(axis=0)
        corr = float(np.corrcoef(ref_prev, our_prev)[0, 1])
        mae = float(np.mean(np.abs(ref_prev - our_prev)))
        # Per-cell agreement broken down by ref label
        both_pos = int(((ours_mat == 1) & (ref_pos == 1)).sum())
        ref_pos_total = int(ref_pos.sum())
        ours_pos_total = int(ours_mat.sum())
        # F1 pooled across all volume-phenotype cells
        tp = both_pos
        fp = int(((ours_mat == 1) & (ref_pos == 0)).sum())
        fn = int(((ours_mat == 0) & (ref_pos == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        summary["variants"][k] = {
            "cell_agreement_rate": float(agree),
            "pooled_precision_vs_haifan": precision,
            "pooled_recall_vs_haifan": recall,
            "pooled_f1_vs_haifan": f1,
            "prevalence_corr_pearson": corr,
            "prevalence_mae": mae,
            "total_positives_ours": ours_pos_total,
            "total_positives_haifan": ref_pos_total,
        }

        # Write per-phenotype comparison
        rows = []
        for name, p in zip(overlap, cols_in_order):
            rp = float(ref_prev[overlap.index(name)])
            op = float(our_prev[overlap.index(name)])
            rows.append(
                {
                    "phenotype": name,
                    "phecode": p,
                    "haifan_prev": rp,
                    "ours_prev": op,
                    "abs_diff": abs(rp - op),
                }
            )
        pd.DataFrame(rows).sort_values("haifan_prev", ascending=False).to_csv(
            OUT / f"compare_{k}_vs_haifan.csv", index=False
        )

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

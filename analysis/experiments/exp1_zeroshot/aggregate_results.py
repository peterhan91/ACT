#!/usr/bin/env python3
"""
Consolidate exp1 zero-shot results across all models + both input protocols.

Reads every <model>/results/summary.csv, emits:
  * results_all.csv  — long-form, every row (model, dataset, mode, n, mean_auc).
  * RESULTS.md       — (A) PRIMARY: each model's OWN published native preprocessing
                       (incl. ours); (B) h5-only controlled-ablation matrix; (C) native
                       reproduction vs the number each paper reports for CT-RATE zero-shot.

Modes: native = the raw scan through the model's OWN published preprocessing + scoring
(the faithful, paper-comparable protocol — the headline); plain = the h5-only controlled
ablation (every model fed the SAME standardized (160,224,224) volume), kept only to show
the representation in isolation. `ours` never runs a separate native pass: the standardized
(160,224,224) HU volume IS our model's own native input, so ours-native == ours-plain.
Native pipelines (checked against each repo): COLIPRI 1mm->384^3->sliding-window 192
(arXiv 2510.15042 A.2; default get_processor's 2mm/single-192 crop is NOT their eval
pipeline and collapses to ~0.49); CT-CLIP data_inference_nii.nii_img_to_tensor
(0.75/0.75/1.5mm, 240x480x480); Merlin MONAI ImageTransforms (RAS/1.5x1.5x3, 224x224x160);
M3D CropForeground+min-max->32x256x256; f-VLM organ-ROI (16/18 labels, native-only).
Native runs on all three datasets (CT-RATE, RAD-ChestCT, RSNA-2023).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_ORDER = ["ours", "colipri", "ctclip", "merlin", "m3dclip", "fvlm"]
DATASET_ORDER = ["ctrate_test", "radchest", "rsna2023_test"]
DLABEL = {"ctrate_test": "CT-RATE (chest, in-domain)",
          "radchest": "RAD-ChestCT (chest, OOD)",
          "rsna2023_test": "RSNA-2023 (abdomen)"}
# Paper-reported CT-RATE zero-shot mean AUROC, for the native reproduction check.
REPORTED = {  # model -> (value-or-None, note)
    "colipri": (0.7981, "arXiv 2510.15042 Table 2 (CRM), 1mm/384/sliding-192, short prompts"),
    "ctclip": (None, "paper reports zero-shot ~0.70 on its 1,304-vol valid set; we score 2,489"),
    "merlin": (None, "abdominal model; no CT-RATE chest zero-shot reported (OOD reference)"),
    "fvlm": (None, "organ-ROI mask-conditioned; 16/18 pathologies; AUC over its 16 covered labels"),
}


def collect() -> pd.DataFrame:
    rows = []
    for csv in sorted(HERE.glob("*/results/summary.csv")):
        df = pd.read_csv(csv); df["_dir"] = csv.parent.parent.name
        rows.append(df)
    if not rows:
        raise SystemExit("no */results/summary.csv found yet")
    out = pd.concat(rows, ignore_index=True)
    out["model"] = out["model"].fillna(out["_dir"])
    return out.drop(columns="_dir")


def matrix(rows, mode, ours_from_plain=False):
    sub = rows[rows["mode"] == mode]
    mat = sub.pivot_table(index="dataset", columns="model", values="mean_auc", aggfunc="last")
    if ours_from_plain:
        # ours has no separate native pass; its (160,224,224) pipeline IS its native
        # input, so surface ours-plain as the ours column of the native matrix.
        op = rows[(rows["model"] == "ours") & (rows["mode"] == "plain")] \
            .set_index("dataset")["mean_auc"]
        mat = mat.reindex(index=sorted(set(mat.index) | set(op.index)))
        mat["ours"] = op.reindex(mat.index)
    mat = mat.reindex(index=[d for d in DATASET_ORDER if d in mat.index])
    cols = [m for m in MODEL_ORDER if m in mat.columns] + \
           [m for m in mat.columns if m not in MODEL_ORDER]
    return mat[cols] if len(mat.columns) else mat


def render(mat):
    if mat.empty or not len(mat.columns):
        return ["_(no rows yet)_"]
    out = ["| dataset | " + " | ".join(mat.columns) + " |",
           "|" + "---|" * (len(mat.columns) + 1)]
    for ds in mat.index:
        vals = mat.loc[ds]; best = vals.max(skipna=True); cells = []
        for m in mat.columns:
            v = vals[m]
            cells.append("—" if pd.isna(v) else (f"**{v:.4f}**" if v == best else f"{v:.4f}"))
        out.append(f"| {DLABEL.get(ds, ds)} | " + " | ".join(cells) + " |")
    return out


def get(rows, model, dataset, mode):
    r = rows[(rows["model"] == model) & (rows["dataset"] == dataset) & (rows["mode"] == mode)]
    return float(r["mean_auc"].iloc[-1]) if len(r) else None


def main():
    rows = collect()
    rows.to_csv(HERE / "results_all.csv", index=False)
    ours_oai = rows[(rows["model"] == "ours") & (rows["mode"] == "openai")] \
        .set_index("dataset")["mean_auc"]

    L = ["# Exp 1 — Zero-shot finding classification: cross-model AUC", "",
         "## A. Native — each model's OWN published preprocessing + scoring (primary)", "",
         "The faithful, paper-comparable protocol: every model ingests the raw scan "
         "through its own published preprocessing + zero-shot scoring (COLIPRI "
         "1mm/384³/sliding-192; CT-CLIP `nii_img_to_tensor`; Merlin MONAI 1.5×1.5×3; "
         "M3D CropForeground; f-VLM organ-ROI). `ours` uses CLEAR-3D's own (160,224,224) "
         "HU pipeline — that *is* our model's native input, so ours-native ≡ ours-plain. "
         "Identical `softmax_pos_neg` scoring. Best per row in **bold** (f-VLM scores only "
         "16/18 labels — see notes — so its cells are not strictly comparable). "
         "`—` = not applicable (f-VLM is chest organ-ROI: no abdomen).", ""]
    L += render(matrix(rows, "native", ours_from_plain=True))
    L += ["", "## B. h5-only controlled ablation (secondary — NOT the headline)", "",
          "Diagnostic only: every model scored on the SAME standardized (160,224,224) "
          "uint8 volume (each baseline just reconstructs HU / renormalizes + resizes to "
          "its grid). Isolates the pretrained representation from preprocessing, so "
          "absolute AUCs differ from the own-pipeline numbers in §A — e.g. CT-CLIP drops "
          "to ~0.52 here purely from the input mismatch, NOT a reproduction gap. The "
          "`ours` row is unchanged (its real pipeline). Best per row in **bold**.", ""]
    L += render(matrix(rows, "plain"))
    L += ["", "## C. Native reproduction vs paper-reported (CT-RATE zero-shot)", "",
          "| model | our native | paper-reported | note |", "|---|---|---|---|"]
    for m in ["colipri", "ctclip", "merlin", "fvlm"]:
        nat = get(rows, m, "ctrate_test", "native")
        rep, note = REPORTED.get(m, (None, ""))
        nat_s = "—" if nat is None else f"{nat:.4f}"
        rep_s = "—" if rep is None else f"{rep:.4f}"
        L.append(f"| {m} | {nat_s} | {rep_s} | {note} |")

    L += ["", "### Notes",
          "- **ours** = CLEAR-3D v1 (dinov2, depth-2); OpenAI-projection mode (reference): "
          + ", ".join(f"{DLABEL.get(d, d)} {ours_oai[d]:.4f}" for d in DATASET_ORDER
                      if d in ours_oai.index) + ".",
          "- **COLIPRI native uses its REAL eval pipeline** (1mm→384³→sliding-window 192, "
          "arXiv 2510.15042). The default `get_processor()` (2mm, single 192³ center crop) is "
          "NOT their eval protocol and collapses to ~0.49; the paper protocol reproduces ~0.80.",
          "- **CT-CLIP native = its own `nii_img_to_tensor`** + present/absent softmax "
          "(`scripts/zero_shot.py`); verified faithful to its forward. Under its own pipeline "
          "it reaches 0.727 on CT-RATE — the ~0.52 in the h5 ablation (§B) was a "
          "preprocessing-mismatch artifact, NOT a reproduction gap. We score our 2,489-vol "
          "ctrate_test, not the paper's 1,304-vol valid set.",
          "- **Merlin** is abdominal — chest sets are OOD; no CT-RATE zero-shot reported.",
          "- **f-VLM** (Alibaba DAMO) = mask-conditioned; pools per-organ TotalSegmentator ROIs "
          "(lung/heart/esophagus/aorta) via query tokens → 16/18 pathologies (drops Medical "
          "material + Lymphadenopathy). Native-only (masks via fvlm/gen_masks.py); its AUC is "
          "over the 16 covered labels, so not directly comparable to the 18-label rows.",
          "- **f-VLM numbers corrected 2026-07-22**: `fvlm/run_zeroshot.py`'s `_score_and_save` "
          "passed the full 18-column label matrix to `save_run` instead of slicing it to the "
          "same 16 `covered` columns used for `probs`, so every archived per-label AUROC was "
          "silently scored against a column-shifted ground truth (off by 1 for the first 5 "
          "labels, by 2 thereafter). Fixed in code and all 3 fvlm `results.json`/summary files "
          "recomputed from the unaffected `probs.npy`.",
          "- n volumes: CT-RATE 2489, RAD-ChestCT 3630, RSNA-2023 943.", ""]
    (HERE / "RESULTS.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\nwrote {HERE/'RESULTS.md'} and {HERE/'results_all.csv'}")


if __name__ == "__main__":
    main()

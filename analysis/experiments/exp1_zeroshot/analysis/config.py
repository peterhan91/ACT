"""
Exp 1 — Automatic concept annotation (zero-shot finding classification).

Single source of truth for the analysis + plotting pipeline:
  * which datasets are in the benchmark,
  * which models/baselines are scored and how they are displayed,
  * which scoring `mode` is canonical for each model,
  * where the per-model result JSONs live and where tables/figures are written.

Protocol (headline): every model ingests the raw scan through its OWN published
pipeline ("native" mode). `ours` (canon v1) is scored in "plain" mode — the
standardized (160,224,224) HU volume IS our model's native input, so ours-plain ==
ours-native. Task = zero-shot finding annotation.
"""
from __future__ import annotations

from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
ANALYSIS_DIR = Path(__file__).resolve().parent          # experiments/exp1_zeroshot/analysis
EXP_DIR = ANALYSIS_DIR.parent                           # experiments/exp1_zeroshot
TABLES_DIR = ANALYSIS_DIR / "tables"                    # aggregated CSV outputs
FIGURES_DIR = ANALYSIS_DIR / "plots" / "figures"        # PNG outputs


def result_json(model_dir: str, dataset: str, mode: str) -> Path:
    """Path to a per-model result JSON: <model>/results/<dataset>__<mode>__results.json"""
    return EXP_DIR / model_dir / "results" / f"{dataset}__{mode}__results.json"


# ── models / baselines ───────────────────────────────────────────────────────
# dir      : folder under experiments/exp1_zeroshot/ holding results/
# display  : ring label in the circular map
# mode     : canonical scoring mode for this model (ours=plain, baselines=native)
# Order here is the *registry* order, NOT the plot order — plot order is the
# CT-RATE ranking computed by aggregate.py (model_ranking.csv).
# dim = native input dimensionality: "3D" volumetric models vs "2D" slice-pooled models.
MODELS = {
    "ours":        dict(display="ACT",          mode="plain",  dim="3D"),
    "merlin":      dict(display="Merlin",       mode="native", dim="3D"),
    "ctclip":      dict(display="CT-CLIP",      mode="native", dim="3D"),
    "fvlm":        dict(display="f-VLM",        mode="native", dim="3D"),
    "m3dclip":     dict(display="M3D-CLIP",     mode="native", dim="3D"),
    "medsiglip":   dict(display="MedSigLIP",    mode="native", dim="2D"),
    "biomedclip":  dict(display="BiomedCLIP",   mode="native", dim="2D"),
    "openai_clip": dict(display="OpenAI CLIP",  mode="native", dim="2D"),
}
# NOTE: COLIPRI (arXiv 2510.15042) is NOT in the requested baseline set, so it is
# excluded here. Its results live under _archive/colipri/ if ever needed.

# Per-model bar colors: soft palette sampled from the reference figure (ours = standout
# magenta, drawn as the rightmost bar; baselines = soft teal/green/orange/purple/gold/blue).
MODEL_COLORS = {
    "ours":        "#C71F73",   # Ours — magenta (standout)
    "merlin":      "#499FA7",   # teal
    "ctclip":      "#ACD79E",   # green
    "fvlm":        "#EEA069",   # orange
    "medsiglip":   "#AE96EA",   # purple
    "biomedclip":  "#F2D282",   # gold
    "openai_clip": "#8FB7DD",   # soft blue
    "m3dclip":     "#F4C9C9",   # pale pink (not in default PLOT_ORDER)
}

# ── datasets ─────────────────────────────────────────────────────────────────
# key      : <dataset> token in the result-JSON filename
# title    : figure title
# region   : chest | abdomen
# groups   : which anatomical category map (in concept_groups.py) to use
RANK_DATASET = "ctrate_test"   # mean AUROC here defines model_ranking.csv (tables only)

# Explicit ring order for the circular maps (outer -> inner), applied to every
# dataset. Overrides the CT-RATE ranking for plotting. Groups ours, then the 3D
# baselines, then the 2D (slice-pooled) baselines. M3D-CLIP is intentionally
# omitted from the figures (still present in the tables). Set to None to fall
# back to the CT-RATE ranking instead.
PLOT_ORDER = ["ours", "merlin", "ctclip", "medsiglip", "biomedclip", "openai_clip"]

# AUROC colorbar: set to a dataset key to draw it on only that figure, or None to draw
# NO colorbar on any plot (panel shares one external legend).
COLORBAR_DATASET = None

# Per-dataset font-size overrides (sparse label sets can afford larger text). Keys are
# plot_dataset() kwargs: concept_label_size / category_label_size / model_label_size.
FONT_OVERRIDES = {}

DATASETS = {
    "ctrate_test":   dict(title="CT-RATE",                region="chest",   groups="chest18"),
    "radchest":      dict(title="RAD-ChestCT",            region="chest",   groups="chest18"),
    "rsna2023_test": dict(title="RSNA Abdominal Trauma",  region="abdomen", groups="rsna9"),
    "pmbb_chest_nc": dict(title="PMBB (chest)",           region="chest",   groups="chest18"),
    "pmbb_abd_ce":   dict(title="PMBB (abdominal)",       region="abdomen", groups="abd30"),
}
# PMBB canonical pools = the SCALED sets (pmbb_chest_nc 9,097 vols / pmbb_abd_ce
# 14,290 vols), which supersede the 2,489-vol pilots (pmbb_chest_test/pmbb_abd_test).

# ── unique-patient counts for the "(N = ... patients)" titles ─────────────────
# Keep these explicit for every dataset used in the four-panel figure: result JSONs
# store the number of scored scans, which differs from the number of unique patients
# for CT-RATE (2,489 scans from 973 patients) and RSNA-2023 (943 scans from 629
# patients). Each PMBB regional pool contains at most one scan per patient.
# Datasets absent here retain the n_volumes fallback for exploratory plots.
PATIENT_COUNTS = {
    "ctrate_test":   973,
    "pmbb_chest_nc": 9_097,
    "pmbb_abd_ce":   14_290,
    "rsna2023_test": 629,
}

# ── composite figures ────────────────────────────────────────────────────────
# One circle merging several datasets' concepts into shared model rings. Each
# member keeps its own concepts/sectors/AUROCs; a colored inner band + legend marks
# which dataset every wedge came from. aggregate.py writes per_label_auc__<name>.csv.
COMPOSITES = {
    "abdominal_combined": dict(
        title="Abdominal concept annotation",
        members=["pmbb_abd_ce", "rsna2023_test"],   # block order, outer arc order
    ),
}
# Per-member dataset-band color + display name (N is filled in at plot time).
DATASET_BAND = {
    "pmbb_abd_ce":   dict(display="PMBB abdominal", color="#4C72B0"),
    "rsna2023_test": dict(display="RSNA 2023",      color="#DD8452"),
}

"""
Centralized paths and label list for the CLEAR-3D port.

Every path can be overridden via environment variables so the same code runs
on a different machine without editing source. Defaults point at our cluster
layout. To run elsewhere, set:

    CLIP3D_CONCEPTS_ROOT  — repo root (defaults to this file's parent)
    CLIP3D_REPO           — path to the clip_3d_ct training code (for clip.tokenize / model loader)
    CLIP3D_EVAL_REPO      — path to clip_3d_eval (for build_and_load helper + configs.json)
    CLIP3D_CONFIGS        — JSON of 3D-CLIP checkpoint configs (defaults to $CLIP3D_EVAL_REPO/configs.json)
    CLIP3D_BEST           — name of the 3D-CLIP config to use as the image encoder
    CT_DATA_ROOT          — root containing the *.h5 volume files
    CT_LABELS_ROOT        — root containing the per-dataset label CSVs
    INSPECT_PHENOTYPE_ROOT — phenotype_labels/per_ct (only needed for exp_phenotype_ct.py)
"""
from __future__ import annotations
import os
from pathlib import Path


def _p(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default))


# ----------------------------------------------------------------------------
# Repo paths needed for imports.
# ----------------------------------------------------------------------------
ROOT = _p("CLIP3D_CONCEPTS_ROOT", str(Path(__file__).resolve().parents[1]))
CLIP3D_REPO = str(_p("CLIP3D_REPO", "/path/to/ACT/model"))
EVAL_REPO = str(_p("CLIP3D_EVAL_REPO", str(ROOT / "eval")))

# ----------------------------------------------------------------------------
# 3D-CLIP backbone we project from. The "best" model on the leaderboard.
# ----------------------------------------------------------------------------
CLIP3D_CONFIGS = str(_p("CLIP3D_CONFIGS", os.path.join(EVAL_REPO, "configs.json")))
CLIP3D_BEST = os.environ.get(
    "CLIP3D_BEST", "clip_3d_ctrate_merlin_v1"
)

# Optional run tag — when set, all outputs and the clip_text portion of the
# concept bank live under outputs/<RUN_TAG>/ and the npz becomes
# `concept_bank.clip_text_emb.<RUN_TAG>.npz`. Lets multiple image encoders
# coexist on disk for side-by-side comparison.
RUN_TAG = os.environ.get("CLIP3D_RUN_TAG", "")

# ----------------------------------------------------------------------------
# Concept bank assets.
# ----------------------------------------------------------------------------
CONCEPT_BANK_PKL = ROOT / "concept_bank.pkl"
_clip_text_name = (
    f"concept_bank.clip_text_emb.{RUN_TAG}.npz" if RUN_TAG
    else "concept_bank.clip_text_emb.npz"
)
CONCEPT_BANK_NPZ = {
    "clip_text": ROOT / _clip_text_name,
    "sfr": ROOT / "concept_bank.sfr_emb.npz",
    "harrier": ROOT / "concept_bank.harrier_emb.npz",
    "openai": ROOT / "concept_bank.openai_emb.npz",
    "f2llm": ROOT / "concept_bank.f2llm_emb.npz",
    "gteqwen2": ROOT / "concept_bank.gteqwen2_emb.npz",
}

_CT_DATA = _p("CT_DATA_ROOT", "/path/to/data_p")
_CT_LABELS = _p("CT_LABELS_ROOT", "/path/to/ACT/model/data")

# ----------------------------------------------------------------------------
# Datasets — h5 + label CSV pairs.  All h5s share `ct_volumes` key, uint8,
# shape (N, 160, 224, 224); they were preprocessed in metadata-row order.
# Some splits have a few metadata rows whose volumes failed to preprocess,
# leaving an h5 shorter than the label CSV — `clear3d.data.load_split` aligns
# by VolumeName and h5 size to handle that.
# ----------------------------------------------------------------------------
DATASETS = {
    "ctrate_test": {
        "h5": str(_CT_DATA / "ctrate_test.h5"),
        "labels_csv": str(_CT_LABELS / "ct_rate/test_predicted_labels.csv"),
        "metadata_csv": str(_CT_LABELS / "ct_rate/test_metadata.csv"),
    },
    "ctrate_train": {
        "h5": str(_CT_DATA / "ctrate_train.h5"),
        "labels_csv": str(_CT_LABELS / "ct_rate/train_predicted_labels.csv"),
        "metadata_csv": str(_CT_LABELS / "ct_rate/train_metadata.csv"),
    },
    "inspect_test": {
        "h5": str(_CT_DATA / "inspect_test.h5"),
        "labels_csv": str(_CT_LABELS / "inspect/test_predicted_labels.csv"),
        "metadata_csv": str(_CT_LABELS / "inspect/test_metadata.csv"),
        "pe_labels_csv": str(_CT_LABELS / "inspect/test_pe_labels.csv"),
    },
    "inspect_train": {
        "h5": str(_CT_DATA / "inspect_train.h5"),
        "labels_csv": str(_CT_LABELS / "inspect/train_predicted_labels.csv"),
        "metadata_csv": str(_CT_LABELS / "inspect/train_metadata.csv"),
        "pe_labels_csv": str(_CT_LABELS / "inspect/train_pe_labels.csv"),
    },
    "inspect_valid": {
        "h5": str(_CT_DATA / "inspect_valid.h5"),
        "labels_csv": str(_CT_LABELS / "inspect/valid_predicted_labels.csv"),
        "metadata_csv": str(_CT_LABELS / "inspect/validation_metadata.csv"),
        "pe_labels_csv": str(_CT_LABELS / "inspect/valid_pe_labels.csv"),
    },
    # RSNA-2023 abdominal trauma. Labels CSV is row-for-row aligned with the
    # paths CSV used to build the h5 (see preprocess_new/generate_label_csvs.py),
    # so h5_idx == csv row index — no VolumeName remapping needed.
    "rsna2023_train": {
        "h5": str(_CT_DATA / "rsna2023_train.h5"),
        "labels_csv": str(_CT_DATA / "rsna2023_train_labels.csv"),
        "kind": "trauma",
    },
    "rsna2023_valid": {
        "h5": str(_CT_DATA / "rsna2023_valid.h5"),
        "labels_csv": str(_CT_DATA / "rsna2023_valid_labels.csv"),
        "kind": "trauma",
    },
    "rsna2023_test": {
        "h5": str(_CT_DATA / "rsna2023_test.h5"),
        "labels_csv": str(_CT_DATA / "rsna2023_test_labels.csv"),
        "kind": "trauma",
    },
    # RSNA-STR PE Detection (kaggle 2020). Labels CSV is row-for-row aligned
    # with the paths CSV used to build the h5 (see
    # preprocess_new/generate_rsna_str_pe_split_csvs.py), so h5_idx == csv
    # row index — same row-aligned loader as rsna2023 (kind: "trauma" is a
    # historical tag for "row-aligned, no VolumeName remap"; we reuse it
    # here rather than introducing a new dispatch value).
    "rsna_str_pe_train": {
        "h5": str(_CT_DATA / "rsna_str_pe_train.h5"),
        "labels_csv": str(_CT_DATA / "rsna_str_pe_train_labels.csv"),
        "kind": "trauma",
    },
    "rsna_str_pe_valid": {
        "h5": str(_CT_DATA / "rsna_str_pe_valid.h5"),
        "labels_csv": str(_CT_DATA / "rsna_str_pe_valid_labels.csv"),
        "kind": "trauma",
    },
    "rsna_str_pe_test": {
        "h5": str(_CT_DATA / "rsna_str_pe_test.h5"),
        "labels_csv": str(_CT_DATA / "rsna_str_pe_test_labels.csv"),
        "kind": "trauma",
    },
    # NSCLC external evals (TCIA LUNG1 / RADIO). One h5 per dataset, covering
    # every patient; per-labelset 80/20 splits live in a separate manifest dir
    # and are joined on PatientID by clear3d.data.load_nsclc_split.
    "lung1": {
        "h5": str(_p("LUNG1_H5", "/path/to/data/tcia/preprocessed/lung1.h5")),
        "paths_csv": str(_p("LUNG1_PATHS", "/path/to/data/tcia/manifests/lung1_paths.csv")),
        "splits_dir": str(_p("NSCLC_MANIFESTS", "/path/to/data/tcia/manifests")),
        "kind": "nsclc",
    },
    "radio": {
        "h5": str(_p("RADIO_H5", "/path/to/data/tcia/preprocessed/radio.h5")),
        "paths_csv": str(_p("RADIO_PATHS", "/path/to/data/tcia/manifests/radio_paths.csv")),
        "splits_dir": str(_p("NSCLC_MANIFESTS", "/path/to/data/tcia/manifests")),
        "kind": "nsclc",
    },
    # RAD-ChestCT (Duke; Draelos et al. 2021) — external OOD diagnosis split.
    # Held out of 3D-CLIP training, and carries the SAME 18 CT-RATE pathology
    # labels (combined_labels.csv), so it reuses LABELS_18 and every cached
    # 18-label prompt embedding. Same h5 preprocessing as all other splits
    # (`ct_volumes`, (N,160,224,224) uint8). The h5 `filenames` (trnXXXXX.npz)
    # align to the label CSV `NoteAcc_DEID` (trnXXXXX) by ID, NOT by row order —
    # see clear3d.data.load_radchest_split.
    "radchest": {
        "h5": str(_p("RADCHEST_H5",
                     "/path/to/ACT/preprocessing/external/radchestct/radchestct.h5")),
        "labels_csv": str(_p("RADCHEST_LABELS",
                             "/path/to/ACT/preprocessing/external/radchestct/combined_labels.csv")),
        "kind": "radchest",
    },
    # PMBB non-contrast retrieval pools (exp2). The manifest CSVs carry an
    # explicit `h5_idx` (original row in pmbb_ct_paths.csv == pmbb_ct_volumes
    # *_iso_spacing*.h5), so the loader is a pure pass-through (kind "pmbb_ret").
    # ISO h5 is used: the non-iso build is mostly zero-padded for these volumes
    # (median 39/160 filled) whereas iso (124/160) matches CT-RATE (110/160).
    "pmbb_chest_test": {
        "h5": str(_p("PMBB_RET_H5",
                     "/path/to/data_p/pmbb_ct_volumes_iso_spacing.h5")),
        "manifest_csv": str(_p("PMBB_CHEST_MANIFEST",
                               str(ROOT / "experiments/exp2_retrieval/pmbb_manifests/pmbb_chest_test.csv"))),
        "kind": "pmbb_ret",
    },
    "pmbb_abd_test": {
        "h5": str(_p("PMBB_RET_H5",
                     "/path/to/data_p/pmbb_ct_volumes_iso_spacing.h5")),
        "manifest_csv": str(_p("PMBB_ABD_MANIFEST",
                               str(ROOT / "experiments/exp2_retrieval/pmbb_manifests/pmbb_abd_test.csv"))),
        "kind": "pmbb_ret",
    },
    # Scaled exp1 classification pools (supersede the 2,489 pilot above): ALL
    # usable volumes, 1/patient. chest is NON-contrast (~9,097); abd is CONTRAST
    # (Merlin-style, ~14,290). Same full ISO h5 + pass-through h5_idx.
    "pmbb_chest_nc": {
        "h5": str(_p("PMBB_RET_H5",
                     "/path/to/data_p/pmbb_ct_volumes_iso_spacing.h5")),
        "manifest_csv": str(_p("PMBB_CHEST_NC_MANIFEST",
                               str(ROOT / "experiments/exp2_retrieval/pmbb_manifests/pmbb_chest_nc.csv"))),
        "kind": "pmbb_ret",
    },
    "pmbb_abd_ce": {
        "h5": str(_p("PMBB_RET_H5",
                     "/path/to/data_p/pmbb_ct_volumes_iso_spacing.h5")),
        "manifest_csv": str(_p("PMBB_ABD_CE_MANIFEST",
                               str(ROOT / "experiments/exp2_retrieval/pmbb_manifests/pmbb_abd_ce.csv"))),
        "kind": "pmbb_ret",
    },
}

# NSCLC labelsets: (dataset, labelset_name, label_columns).
# Built by preprocess_new/generate_nsclc_split_csvs.py.
NSCLC_LABELSETS = [
    ("lung1", "os2yr",          ["os2yr"]),
    ("lung1", "histology",      ["adenocarcinoma", "squamous_cell_carcinoma",
                                  "large_cell", "nos"]),
    ("lung1", "stage_advanced", ["stage_advanced"]),
    ("radio", "egfr",             ["egfr"]),
    ("radio", "kras",             ["kras"]),
    ("radio", "recurrence",       ["recurrence"]),
    ("radio", "pleural_invasion", ["pleural_invasion"]),
    ("radio", "os2yr",            ["os2yr"]),
]

# Human-readable prompt strings per NSCLC labelset (positive-class wording).
# Used by precompute_openai_labels.py and any downstream zero-shot driver.
# Lists are aligned column-for-column with NSCLC_LABELSETS above.
NSCLC_LABELSET_PROMPTS = {
    "os2yr":            ["death within two years of diagnosis"],
    "histology":        ["lung adenocarcinoma",
                         "lung squamous cell carcinoma",
                         "large cell lung carcinoma",
                         "non-small cell lung carcinoma not otherwise specified"],
    "stage_advanced":   ["advanced stage non-small cell lung cancer"],
    "egfr":             ["EGFR mutated lung cancer"],
    "kras":             ["KRAS mutated lung cancer"],
    "recurrence":       ["lung cancer recurrence after treatment"],
    "pleural_invasion": ["pleural invasion by lung tumor"],
}

# Same 18 CT-RATE pathology labels eval_all.py uses.  INSPECT predicted-label
# CSV has the identical schema (predicted by the same RadBERT-style model on
# INSPECT reports), so we can reuse this list for external validation.
# INSPECT's 3 native PE-related task labels (column names differ between
# train/test which use snake_case and valid which uses verbose strings, so we
# canonicalize to the snake_case set; data.load_pe_split renames as needed).
LABELS_PE = ["pe_positive", "pe_acute", "pe_subsegmentalonly"]

LABELS_PE_VERBOSE_MAP = {
    "Pulmonary embolism": "pe_positive",
    "Acute pulmonary embolism": "pe_acute",
    "Subsegmental pulmonary embolism": "pe_subsegmentalonly",
}

LABELS_18 = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]

# RSNA-2023 abdominal trauma: 8 organ-injury binaries + any_injury meta-label.
# Healthy columns (bowel_healthy etc.) are derivable as 1-(low+high) for the
# 3-way organs and 1-injury for the 2-way organs, so we drop them from the
# eval target list. Order matches the prompt list `LABELS_RSNA2023_TRAUMA_PROMPTS`
# below — keep them in sync.
LABELS_RSNA2023_TRAUMA = [
    "bowel_injury",
    "extravasation_injury",
    "kidney_low",
    "kidney_high",
    "liver_low",
    "liver_high",
    "spleen_low",
    "spleen_high",
    "any_injury",
]
LABELS_RSNA2023_TRAUMA_PROMPTS = [
    "bowel injury",
    "active extravasation",
    "low-grade kidney injury",
    "high-grade kidney injury",
    "low-grade liver injury",
    "high-grade liver injury",
    "low-grade spleen injury",
    "high-grade spleen injury",
    "any abdominal injury",
]

# RSNA-STR PE Detection (kaggle 2020). 10 study-level binary labels selected
# as primary eval targets. Dropped from this list (but still present in the
# labels CSV for ad-hoc analysis): true_filling_defect_not_pe,
# qa_motion / qa_contrast / flow_artifact (QA flags, not pathology). Order
# matches LABELS_RSNA_STR_PE_PROMPTS below — keep them in sync.
#
# Note on `negative_exam_for_pe` and `indeterminate`: these are kaggle
# study-level exam-status flags (not strictly pathology). The pos/neg
# prompt convention (`f"no {prompt}"`) produces awkward double-negatives
# for them in zero-shot, but the linear/CBM probes use the image
# representation directly and are unaffected.
LABELS_RSNA_STR_PE = [
    "pe_present_on_study",
    "negative_exam_for_pe",
    "indeterminate",
    "rv_lv_ratio_gte_1",
    "rv_lv_ratio_lt_1",
    "leftsided_pe",
    "rightsided_pe",
    "central_pe",
    "acute_and_chronic_pe",
    "chronic_pe",
]
LABELS_RSNA_STR_PE_PROMPTS = [
    "pulmonary embolism",
    "negative exam for pulmonary embolism",
    "indeterminate exam for pulmonary embolism",
    "right-to-left ventricular ratio at least 1",
    "right-to-left ventricular ratio less than 1",
    "left-sided pulmonary embolism",
    "right-sided pulmonary embolism",
    "central pulmonary embolism",
    "acute on chronic pulmonary embolism",
    "chronic pulmonary embolism",
]

# ----------------------------------------------------------------------------
# Phenotype labels (phecode-derived, INSPECT-only) — peterhan91/phenotype_labels.
# 22461 CT volumes × 1692 phecodes; manifest defines train/valid/test splits
# that are slightly different from the existing inspect_* h5 splits. We pull
# img_feats / llm_repr by VolumeName across the three inspect_* caches.
# ----------------------------------------------------------------------------
PHENOTYPE_ROOT = _p("INSPECT_PHENOTYPE_ROOT", "/path/to/phenotype_labels/per_ct")
PHENOTYPE_LABELS = PHENOTYPE_ROOT / "per_ct_labels_visit_only.parquet"
PHENOTYPE_MANIFEST = PHENOTYPE_ROOT / "manifest.parquet"
PHENOTYPE_NAMES_CSV = PHENOTYPE_ROOT / "phenotypes.csv"

# ----------------------------------------------------------------------------
# INSPECT official prognosis labels (Stanford INSPECT v2 release).
# `labels_20250611.tsv` is keyed by impression_id and contains binary outcome
# flags + time-to-event pairs. We join it onto the inspect_{train,valid,test}
# caches via the impression_id column in *_metadata.csv.
# ----------------------------------------------------------------------------
INSPECT_OFFICIAL_LABELS_TSV = _p(
    "INSPECT_OFFICIAL_LABELS_TSV",
    "/path/to/data/Inspect_v2.0/"
    "inspectamultimodaldatasetforpulmonaryembolismdiagnosisandprog-3/full/"
    "labels_20250611.tsv",
)
LABELS_INSPECT_PROGNOSIS = [
    "1_month_mortality", "6_month_mortality", "12_month_mortality",
    "1_month_readmission", "6_month_readmission", "12_month_readmission",
    "12_month_PH",
]
# Human-readable prompt strings (one per column in LABELS_INSPECT_PROGNOSIS,
# same order). Pos/neg prompts are `<prompt>` and `f"no {prompt}"` — matches
# the convention used by precompute_openai_labels.py for every other labelset.
LABELS_INSPECT_PROGNOSIS_PROMPTS = [
    "death within 1 month",
    "death within 6 months",
    "death within 12 months",
    "hospital readmission within 1 month",
    "hospital readmission within 6 months",
    "hospital readmission within 12 months",
    "pulmonary hypertension within 12 months",
]

# ----------------------------------------------------------------------------
# Output dirs.
# ----------------------------------------------------------------------------
OUT = (ROOT / "outputs" / RUN_TAG) if RUN_TAG else (ROOT / "outputs")
CACHE = OUT / "cache"
ZEROSHOT_DIR = OUT / "zeroshot"
CBM_DIR = OUT / "cbm"
AUDIT_DIR = OUT / "audit"
EXTERNAL_DIR = OUT / "external"
LOG_DIR = ROOT / "logs"

for _d in (CACHE, ZEROSHOT_DIR, CBM_DIR, AUDIT_DIR, EXTERNAL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def cache_img_feats(dataset: str) -> Path:
    return CACHE / f"img_feats.{dataset}.npy"


def cache_volume_index(dataset: str) -> Path:
    """Aligned VolumeName + h5_idx ordering for `dataset`. Drives label lookup."""
    return CACHE / f"volume_index.{dataset}.csv"


def cache_llm_repr(dataset: str, llm: str) -> Path:
    """Image embeddings projected to LLM concept space, normalized."""
    return CACHE / f"llm_repr.{dataset}.{llm}.npy"


def cache_label_emb(llm: str) -> Path:
    """LLM (pos, neg) prompt embeddings for the 18 labels."""
    return CACHE / f"label_emb.{llm}.npz"

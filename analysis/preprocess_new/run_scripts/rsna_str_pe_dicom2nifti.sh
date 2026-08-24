#!/bin/bash
#SBATCH --job-name=RSNA_STR_PE_DICOM2NIfTI
#SBATCH --output=/path/to/logs/rsna_str_pe_dicom2nifti_%j.out
#SBATCH --error=/path/to/logs/rsna_str_pe_dicom2nifti_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --partition=all
#SBATCH --mail-type=END,FAIL

# Step 1 of RSNA-STR PE Detection preprocessing.
#   1a. DICOM series -> NIfTI conversion of the kaggle TRAIN split
#       (~7,279 studies / examinations). Each series becomes
#       <Study>__<Series>.nii.gz.
#   1b. Study-wise 80/10/10 split (seed 42) with hard overlap check,
#       and per-split paths.csv + labels.csv (study-level labels —
#       pe_present_on_study = any(pe_present_on_image) across slices).
#
# NOTE on splitting key. Kaggle's anonymization strips every patient tag
# from the DICOMs (no PatientID, PatientName, AccessionNumber,
# InstitutionName — verified empirically). Per Colak et al. Radiology AI
# 2021, the dataset is de-duplicated to one study per patient *per site*
# (n=129 excluded), so splitting on StudyInstanceUID is patient-wise
# within each site. Cross-site duplication (same patient at multiple of
# the 5 source sites) is not addressed in the paper and cannot be
# detected here. See generate_rsna_str_pe_split_csvs.py docstring.
#
# Kaggle's competition test/ DICOMs are unlabeled (private leaderboard)
# and are dropped.
#
# Outputs:
#   /path/to/data_p/rsna_str_pe_nifti/<Study>__<Series>.nii.gz
#   /path/to/data_p/rsna_str_pe_nifti/_paths.csv
#   /path/to/data_p/rsna_str_pe_{train,valid,test}_paths.csv
#   /path/to/data_p/rsna_str_pe_{train,valid,test}_labels.csv
#
# Next: sbatch rsna_str_pe_preprocess_all.sh

set -euo pipefail

source /path/to/miniconda3/etc/profile.d/conda.sh
# Use full env path: `conda activate ctproject` may resolve to a different
# (incomplete) env under ~/.conda/envs/ if one exists there.
conda activate /path/to/miniconda3/envs/ctproject

REPO="${ACT_REPO:-/path/to/ACT}/analysis/preprocess_new"
HELPER_DICOM="${ACT_REPO:-/path/to/ACT}/analysis/dropbox_patches/dicom_series_to_nifti.py"
HELPER_SPLIT="$REPO/generate_rsna_str_pe_split_csvs.py"

KAGGLE_ROOT="/path/to/data/competitions/rsna-str-pulmonary-embolism-detection"
INPUT_DIR="$KAGGLE_ROOT/train"
LABELS_SRC="$KAGGLE_ROOT/train.csv"

OUTPUT_DIR="/path/to/data_p/rsna_str_pe_nifti"
MASTER_CSV="$OUTPUT_DIR/_paths.csv"
SPLIT_DIR="/path/to/data_p"

mkdir -p "$OUTPUT_DIR" "$SPLIT_DIR"

echo "=== [$(date)] Step 1a: DICOM -> NIfTI ==="
python "$HELPER_DICOM" \
    --input_dir   "$INPUT_DIR" \
    --output_dir  "$OUTPUT_DIR" \
    --paths_csv   "$MASTER_CSV" \
    --min_slices  10 \
    --num_workers 32

echo
echo "=== [$(date)] Step 1b: study-wise split + study-level labels ==="
python "$HELPER_SPLIT" \
    --paths_csv  "$MASTER_CSV" \
    --labels_csv "$LABELS_SRC" \
    --out_dir    "$SPLIT_DIR" \
    --seed       42 \
    --ratios     0.8 0.1 0.1

echo
echo "RSNA-STR PE step 1 complete."
echo "  NIfTI dir       : $OUTPUT_DIR"
echo "  Split path CSVs : $SPLIT_DIR/rsna_str_pe_{train,valid,test}_paths.csv"
echo "  Split label CSVs: $SPLIT_DIR/rsna_str_pe_{train,valid,test}_labels.csv"
echo "Next: sbatch rsna_str_pe_preprocess_all.sh"

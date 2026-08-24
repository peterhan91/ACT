#!/bin/bash
#SBATCH --job-name=RSNA2023_DICOM2NIfTI
#SBATCH --output=/path/to/logs/rsna2023_dicom2nifti_%j.out
#SBATCH --error=/path/to/logs/rsna2023_dicom2nifti_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=all
#SBATCH --mail-type=END,FAIL

# Step 1 of RSNA-2023 preprocessing: DICOM-series -> NIfTI conversion of the
# kaggle TRAIN split (~3,147 patients, ~1.5M DICOMs across 4,711 series), then
# patient-wise 70/10/20 split and per-split label CSV generation.
# Kaggle's TEST labels are private, so we repurpose kaggle-train as our
# train + valid + test universe.

set -euo pipefail

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

REPO="${ACT_REPO:-/path/to/ACT}/analysis/preprocess_new"
HELPER="${ACT_REPO:-/path/to/ACT}/analysis/dropbox_patches/dicom_series_to_nifti.py"
# NOTE: generate_rsna2023_split_csvs.py was not retained. The seed-42 patient
# split it produced is deterministically reconstructed and verified by
# analysis/experiments/supplementary/reconstruct_rsna2023_test_manifest.py.
SPLITTER="${ACT_REPO:-/path/to/ACT}/analysis/dropbox_patches/generate_rsna2023_split_csvs.py"
LABEL_GEN="$REPO/generate_label_csvs.py"

INPUT_DIR="/path/to/data/RSNA2023/train_images"
OUTPUT_DIR="/path/to/data_p/rsna2023_nifti"
MASTER_CSV="$OUTPUT_DIR/_paths.csv"
SPLIT_DIR="/path/to/data_p"
LABELS_SRC="/path/to/data/RSNA2023/train_2024.csv"

mkdir -p "$OUTPUT_DIR" "$SPLIT_DIR"

echo "=== [$(date)] Step 1a: DICOM -> NIfTI ==="
python "$HELPER" \
    --input_dir   "$INPUT_DIR" \
    --output_dir  "$OUTPUT_DIR" \
    --paths_csv   "$MASTER_CSV" \
    --min_slices  10 \
    --num_workers 32

echo
echo "=== [$(date)] Step 1b: 70/10/20 patient-wise split CSVs ==="
if [[ -f "$SPLITTER" ]]; then
    python "$SPLITTER" \
        --paths_csv  "$MASTER_CSV" \
        --out_dir    "$SPLIT_DIR" \
        --labels_csv "$LABELS_SRC" \
        --seed       42 \
        --ratios     0.7 0.1 0.2
else
    echo "[skip] $SPLITTER not retained; reconstruct the seed-42 split with"
    echo "       analysis/experiments/supplementary/reconstruct_rsna2023_test_manifest.py"
fi

echo
echo "=== [$(date)] Step 1c: per-split label CSVs ==="
for SPLIT in train valid test; do
    python "$LABEL_GEN" \
        --dataset           rsna2023 \
        --paths_csv         "$SPLIT_DIR/rsna2023_${SPLIT}_paths.csv" \
        --source_labels_csv "$LABELS_SRC" \
        --out_csv           "$SPLIT_DIR/rsna2023_${SPLIT}_labels.csv"
done

echo
echo "RSNA-2023 step 1 complete."
echo "  NIfTI dir       : $OUTPUT_DIR"
echo "  Split path CSVs : $SPLIT_DIR/rsna2023_{train,valid,test}_paths.csv"
echo "  Split label CSVs: $SPLIT_DIR/rsna2023_{train,valid,test}_labels.csv"
echo "Next: sbatch rsna2023_preprocess_all.sh"

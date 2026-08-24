#!/bin/bash
#SBATCH --job-name=MMPE_NPY2NIfTI
#SBATCH --output=/path/to/logs/mmpe_npy2nifti_%j.out
#SBATCH --error=/path/to/logs/mmpe_npy2nifti_%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --mail-type=END,FAIL

# Step 1 of MultimodalPE preprocessing: .npy -> .nii.gz, then write split path
# CSVs using the official Huang 2020 split (1454 train / 193 val / 190 test).

set -euo pipefail

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

REPO="${ACT_REPO:-/path/to/ACT}/analysis/preprocess_new"
INPUT_DIR="/path/to/data/MultimodalPE/multimodalpulmonaryembolismdataset"
OUTPUT_DIR="/path/to/data_p/multimodalpe_nifti"
MASTER_CSV="$OUTPUT_DIR/_paths.csv"
SPLIT_DIR="/path/to/data_p"
LABELS_SRC="$INPUT_DIR/Labels.csv"

mkdir -p "$OUTPUT_DIR" "$SPLIT_DIR"

echo "=== [$(date)] Step 1a: .npy -> .nii.gz ==="
python "$REPO/multimodalpe_npy2nifti.py" \
    --input_dir   "$INPUT_DIR" \
    --output_dir  "$OUTPUT_DIR" \
    --paths_csv   "$MASTER_CSV" \
    --num_workers 16

echo
echo "=== [$(date)] Step 1b: official-split path CSVs ==="
python "$REPO/generate_multimodalpe_split_csvs.py" \
    --paths_csv  "$MASTER_CSV" \
    --labels_csv "$LABELS_SRC" \
    --out_dir    "$SPLIT_DIR"

echo
echo "=== [$(date)] Step 1c: per-split label CSVs ==="
for SPLIT in train valid test; do
    python "$REPO/generate_label_csvs.py" \
        --dataset            multimodalpe \
        --paths_csv          "$SPLIT_DIR/multimodalpe_${SPLIT}_paths.csv" \
        --source_labels_csv  "$LABELS_SRC" \
        --out_csv            "$SPLIT_DIR/multimodalpe_${SPLIT}_labels.csv"
done

echo
echo "MultimodalPE step 1 complete."
echo "  NIfTI dir       : $OUTPUT_DIR"
echo "  Split path CSVs : $SPLIT_DIR/multimodalpe_{train,valid,test}_paths.csv"
echo "  Split label CSVs: $SPLIT_DIR/multimodalpe_{train,valid,test}_labels.csv"
echo "Next: sbatch multimodalpe_preprocess_all.sh"

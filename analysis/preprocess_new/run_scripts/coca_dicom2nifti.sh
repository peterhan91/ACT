#!/bin/bash
#SBATCH --job-name=COCA_DICOM2NIfTI
#SBATCH --output=/path/to/logs/coca_dicom2nifti_%j.out
#SBATCH --error=/path/to/logs/coca_dicom2nifti_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --mail-type=END,FAIL

# Step 1 of COCA preprocessing: DICOM-series -> NIfTI conversion (gated + nongated),
# patient-wise 70/10/20 split (default mode unifies pid namespace because
# 211/214 nongated PIDs collide numerically with gated PIDs), and label CSVs
# (Agatston scoring from calcium_xml for gated patients; nongated -> NaN).

set -euo pipefail

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

REPO="${ACT_REPO:-/path/to/ACT}/analysis/preprocess_new"
HELPER="${ACT_REPO:-/path/to/ACT}/analysis/dropbox_patches/dicom_series_to_nifti.py"
SPLITTER="${ACT_REPO:-/path/to/ACT}/analysis/dropbox_patches/generate_coca_split_csvs.py"
LABEL_GEN="$REPO/generate_label_csvs.py"

INPUT_DIR="/path/to/data/COCA/cocacoronarycalciumandchestcts-2"
OUTPUT_DIR="/path/to/data_p/coca_nifti"
MASTER_CSV="$OUTPUT_DIR/_paths.csv"
SPLIT_DIR="/path/to/data_p"
COCA_XML_DIR="$INPUT_DIR/Gated_release_final/calcium_xml"

mkdir -p "$OUTPUT_DIR" "$SPLIT_DIR"

echo "=== [$(date)] Step 1a: DICOM -> NIfTI ==="
python "$HELPER" \
    --input_dir   "$INPUT_DIR" \
    --output_dir  "$OUTPUT_DIR" \
    --paths_csv   "$MASTER_CSV" \
    --min_slices  10 \
    --num_workers 16

echo
echo "=== [$(date)] Step 1b: 70/10/20 patient-wise split CSVs ==="
python "$SPLITTER" \
    --paths_csv "$MASTER_CSV" \
    --out_dir   "$SPLIT_DIR" \
    --seed      42 \
    --ratios    0.7 0.1 0.2

echo
echo "=== [$(date)] Step 1c: per-split label CSVs (Agatston from XML) ==="
for SPLIT in train valid test; do
    python "$LABEL_GEN" \
        --dataset       coca \
        --paths_csv     "$SPLIT_DIR/coca_${SPLIT}_paths.csv" \
        --coca_xml_dir  "$COCA_XML_DIR" \
        --out_csv       "$SPLIT_DIR/coca_${SPLIT}_labels.csv"
done

echo
echo "COCA step 1 complete."
echo "  NIfTI dir       : $OUTPUT_DIR"
echo "  Split path CSVs : $SPLIT_DIR/coca_{train,valid,test}_paths.csv"
echo "  Split label CSVs: $SPLIT_DIR/coca_{train,valid,test}_labels.csv"
echo "Next: sbatch coca_preprocess_all.sh"

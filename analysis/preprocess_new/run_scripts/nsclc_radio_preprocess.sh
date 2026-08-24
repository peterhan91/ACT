#!/bin/bash
#SBATCH --job-name=NSCLC_RADIO
#SBATCH --output=/path/to/logs/nsclc_radio_%j.out
#SBATCH --error=/path/to/logs/nsclc_radio_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL

# RADIO (NSCLC-Radiogenomics) preprocessing — pick canonical diagnostic CT
# per patient (212 candidates; series include CT, PET, fusion, scouts),
# convert to NIfTI, build a 160×224×224 h5.

set -euo pipefail
ulimit -t unlimited

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/miniconda3/envs/ctproject

REPO=${ACT_REPO:-/path/to/ACT}/analysis/preprocess_new
CLIP_REPO=${ACT_REPO:-/path/to/ACT}/preprocessing

DICOM=/path/to/data/tcia/nsclc_radiogenomics/dicom
NIFTI=/path/to/data/tcia/nsclc_radiogenomics/nifti_canonical
PATHS=/path/to/data/tcia/manifests/radio_paths.csv
H5=/path/to/data/tcia/preprocessed/radio.h5

mkdir -p "$NIFTI" "$(dirname "$PATHS")" "$(dirname "$H5")"

echo "=== [$(date)] Step 1: pick canonical CT + DICOM→NIfTI ==="
python "$REPO/nsclc_pick_and_convert.py" \
    --input_dir   "$DICOM" \
    --output_dir  "$NIFTI" \
    --paths_csv   "$PATHS" \
    --num_workers 64

echo "=== [$(date)] Step 2: NIfTI→H5 ==="
cd "$CLIP_REPO"
python run_preprocess.py \
    --ct_data_path "$PATHS" \
    --ct_out_path  "$H5" \
    --target_shape 160 224 224 \
    --num_workers  16

echo "RADIO preprocessing complete."
ls -la "$H5" "$PATHS"

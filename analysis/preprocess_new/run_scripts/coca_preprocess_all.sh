#!/bin/bash
#SBATCH --job-name=COCA_Preprocess_All
#SBATCH --output=/path/to/logs/coca_preprocess_all_%j.out
#SBATCH --error=/path/to/logs/coca_preprocess_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=all
#SBATCH --mail-type=END,FAIL

# Step 2 of COCA preprocessing: NIfTI -> 3 HDF5 files (train/valid/test).
# Identical pipeline to ctrate / inspect / merlin / mmpe / rsna2023.
#
# Prereq: coca_dicom2nifti.sh produced the path CSVs.

set -euo pipefail

# Lift the per-process CPU-time rlimit. With --cpus-per-task=32 and the master
# python driving 32 worker processes, accumulated CPU-seconds blow past the
# default RLIMIT_CPU well before wall time runs out. Without this, runs of
# >~1500 volumes get SIGXCPU mid-batch (observed at MMPE batch 37/46).
ulimit -t unlimited

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

REPO_PATH="${ACT_REPO:-/path/to/ACT}/preprocessing"
SPLIT_DIR="/path/to/data_p"
OUT_DIR="/path/to/data_p"

mkdir -p "$OUT_DIR"
cd "$REPO_PATH"

for SPLIT in train valid test; do
    PATHS_CSV="$SPLIT_DIR/coca_${SPLIT}_paths.csv"
    if [ ! -f "$PATHS_CSV" ]; then
        echo "ERROR: $PATHS_CSV not found. Run coca_dicom2nifti.sh first." >&2
        exit 1
    fi
    echo "=== [$(date)] COCA $SPLIT ==="
    python run_preprocess.py \
        --ct_data_path "$PATHS_CSV" \
        --ct_out_path  "$OUT_DIR/coca_${SPLIT}.h5" \
        --target_shape 160 224 224 \
        --num_workers  32
done

echo
echo "COCA step 2 complete."
ls -la "$OUT_DIR"/coca_{train,valid,test}.h5 2>/dev/null

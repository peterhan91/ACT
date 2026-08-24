#!/bin/bash
#SBATCH --job-name=RSNA2023_Preprocess_All
#SBATCH --output=/path/to/logs/rsna2023_preprocess_all_%j.out
#SBATCH --error=/path/to/logs/rsna2023_preprocess_all_%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=long
#SBATCH --mail-type=END,FAIL

# Step 2 of RSNA-2023 preprocessing: NIfTI -> 3 HDF5 files (train/valid/test).
# Identical pipeline to ctrate / inspect / merlin / mmpe.
#
# Prereq: rsna2023_dicom2nifti.sh produced the path CSVs.

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
    PATHS_CSV="$SPLIT_DIR/rsna2023_${SPLIT}_paths.csv"
    if [ ! -f "$PATHS_CSV" ]; then
        echo "ERROR: $PATHS_CSV not found. Run rsna2023_dicom2nifti.sh first." >&2
        exit 1
    fi
    echo "=== [$(date)] RSNA-2023 $SPLIT ==="
    python run_preprocess.py \
        --ct_data_path "$PATHS_CSV" \
        --ct_out_path  "$OUT_DIR/rsna2023_${SPLIT}.h5" \
        --target_shape 160 224 224 \
        --num_workers  32
done

echo
echo "RSNA-2023 step 2 complete."
ls -la "$OUT_DIR"/rsna2023_{train,valid,test}.h5 2>/dev/null

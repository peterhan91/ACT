#!/bin/bash
#SBATCH --job-name=RSNA_STR_PE_Preprocess_All
#SBATCH --output=/path/to/logs/rsna_str_pe_preprocess_all_%j.out
#SBATCH --error=/path/to/logs/rsna_str_pe_preprocess_all_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --partition=all
#SBATCH --mail-type=END,FAIL

# Step 2 of RSNA-STR PE preprocessing: NIfTI -> 3 HDF5 files
# (train/valid/test). Identical pipeline to ctrate / inspect / merlin /
# mmpe / rsna2023. Expects ~7,279 series across the three splits, ~50%
# more than rsna2023 — walltime bumped to 120h to be safe.
#
# Prereq: rsna_str_pe_dicom2nifti.sh produced the path CSVs.

set -euo pipefail

# Lift the per-process CPU-time rlimit. With --cpus-per-task=32 and the
# master python driving 32 worker processes, accumulated CPU-seconds blow
# past the default RLIMIT_CPU well before wall time runs out. Without
# this, runs of >~1500 volumes get SIGXCPU mid-batch (observed at MMPE
# batch 37/46).
ulimit -t unlimited

source /path/to/miniconda3/etc/profile.d/conda.sh
# Use full env path: `conda activate ctproject` may resolve to a different
# (incomplete) env under ~/.conda/envs/ if one exists there.
conda activate /path/to/miniconda3/envs/ctproject

REPO_PATH="${ACT_REPO:-/path/to/ACT}/preprocessing"
SPLIT_DIR="/path/to/data_p"
OUT_DIR="/path/to/data_p"

mkdir -p "$OUT_DIR"
cd "$REPO_PATH"

for SPLIT in train valid test; do
    PATHS_CSV="$SPLIT_DIR/rsna_str_pe_${SPLIT}_paths.csv"
    if [ ! -f "$PATHS_CSV" ]; then
        echo "ERROR: $PATHS_CSV not found. Run rsna_str_pe_dicom2nifti.sh first." >&2
        exit 1
    fi
    echo "=== [$(date)] RSNA-STR PE $SPLIT ==="
    python run_preprocess.py \
        --ct_data_path "$PATHS_CSV" \
        --ct_out_path  "$OUT_DIR/rsna_str_pe_${SPLIT}.h5" \
        --target_shape 160 224 224 \
        --num_workers  32
done

echo
echo "RSNA-STR PE step 2 complete."
ls -la "$OUT_DIR"/rsna_str_pe_{train,valid,test}.h5 2>/dev/null

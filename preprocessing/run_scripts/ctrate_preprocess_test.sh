#!/bin/bash
#SBATCH --job-name=CTRate_Preprocess_Test
#SBATCH --output=ctrate_test_preprocess_%j.out
#SBATCH --error=ctrate_test_preprocess_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --mail-type=END,FAIL

# Load conda environment
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
REPO_PATH="${ACT_REPO:-/path/to/ACT}/preprocessing"
OUTPUT_PATH="/path/to/data/ctrate_test.h5"

# Create output directory if it doesn't exist
mkdir -p /path/to/data/

# Run preprocessing with split-based approach
cd $REPO_PATH
python run_preprocess.py \
    --split test \
    --ct_out_path "$OUTPUT_PATH" \
    --target_shape 160 224 224 \
    --num_workers 16

echo "CT-RATE test preprocessing completed successfully!"
echo "Output saved to: $OUTPUT_PATH"
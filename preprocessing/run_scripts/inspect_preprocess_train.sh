#!/bin/bash
#SBATCH --job-name=Inspect_Preprocess_Train
#SBATCH --output=inspect_train_preprocess_%j.out
#SBATCH --error=inspect_train_preprocess_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=long
#SBATCH --mail-type=END,FAIL

# Load conda environment
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
REPO_PATH="${ACT_REPO:-/path/to/ACT}/preprocessing"
OUTPUT_PATH="/path/to/data/inspect_train.h5"

# Create output directory if it doesn't exist
mkdir -p /path/to/data/

# Run preprocessing with Inspect dataset
cd $REPO_PATH
python run_preprocess.py \
    --dataset inspect \
    --split train \
    --ct_out_path "$OUTPUT_PATH" \
    --target_shape 160 224 224 \
    --num_workers 32

echo "Inspect train preprocessing completed successfully!"
echo "Output saved to: $OUTPUT_PATH"
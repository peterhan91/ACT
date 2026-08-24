#!/bin/bash
#SBATCH --job-name=Merlin_Preprocess_Combined
#SBATCH --output=merlin_combined_preprocess_%j.out
#SBATCH --error=merlin_combined_preprocess_%j.err
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
OUTPUT_PATH="/path/to/data/merlin_train.h5"

# Create output directory if it doesn't exist
mkdir -p /path/to/data/

# First, generate combined CSV files from all splits
cd $REPO_PATH/run_scripts
python -c "
import pandas as pd
import os

# Read all split CSVs
train_df = pd.read_csv('merlin_train_paths.csv')
valid_df = pd.read_csv('merlin_valid_paths.csv')
test_df = pd.read_csv('merlin_test_paths.csv')

# Combine all paths
combined_paths = pd.concat([train_df, valid_df, test_df], ignore_index=True)
combined_paths.to_csv('merlin_combined_paths.csv', index=False)

print(f'Combined {len(train_df)} train + {len(valid_df)} valid + {len(test_df)} test = {len(combined_paths)} total paths')
"

# Run preprocessing with combined Merlin dataset
cd $REPO_PATH
python run_preprocess.py \
    --dataset merlin \
    --ct_data_path "$REPO_PATH/run_scripts/merlin_combined_paths.csv" \
    --ct_out_path "$OUTPUT_PATH" \
    --metadata_path "$REPO_PATH/data/merlin/train_metadata.csv" \
    --target_shape 160 224 224 \
    --num_workers 32

echo "Merlin combined preprocessing completed successfully!"
echo "Output saved to: $OUTPUT_PATH"
echo "This file contains all Merlin samples (train+valid+test) for CLIP training."
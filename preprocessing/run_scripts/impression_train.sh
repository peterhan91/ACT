#!/bin/bash
#SBATCH --job-name=Impression_Train
#SBATCH --output=impression_train_%j.out
#SBATCH --error=impression_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=80G
#SBATCH --partition=ai
#SBATCH --mail-type=END,FAIL

# Load CUDA module
module load cuda/12.4

# Load conda environment
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
REPO_PATH="${ACT_REPO:-/path/to/ACT}/preprocessing"
TARGET_FILE="${ACT_REPO:-/path/to/ACT}/model/data/ct_rate/train_reports.csv"
OUTPUT_FILE="/path/to/data/train_reports.json"

# Create output directory if it doesn't exist
mkdir -p /path/to/data/

# Change to repository directory
cd $REPO_PATH

# Run impression generation for training set
python impression_section.py \
    --target_file "$TARGET_FILE" \
    --output_file "$OUTPUT_FILE" \
    --num_examples 8 \
    --max_new_tokens 8192

echo "Train impression generation completed!"
echo "Updated CSV saved to: ${OUTPUT_FILE%.json}.csv"
echo "Generation log saved to: $OUTPUT_FILE"
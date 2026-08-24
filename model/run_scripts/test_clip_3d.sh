#!/bin/bash
#SBATCH --job-name=CLIP_3D_Test
#SBATCH --output=clip_3d_test_%j.out
#SBATCH --error=clip_3d_test_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
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
REPO_PATH="${ACT_REPO:-/path/to/ACT}/model"

# Model path - update this to your trained model
MODEL_PATH="/path/to/models/clip_3d/clip_3d_ct_v2_ddp/best_model.pt"
MODEL_NAME="clip_3d_ct_v2_ddp_test"

# CT-RATE test paths
CTRATE_TEST_CT="/path/to/data/ctrate_test.h5"
CTRATE_TEST_LABELS="${ACT_REPO:-/path/to/ACT}/model/data/ct_rate/test_predicted_labels.csv"


# Results directory
RESULTS_DIR="/path/to/results/clip_3d/"

# Create results directory
mkdir -p $RESULTS_DIR

# Change to repository directory
cd $REPO_PATH

echo "Starting CLIP-3D-CT model testing..."
echo "Model: $MODEL_PATH"
echo "Results will be saved to: $RESULTS_DIR"
echo ""
echo "Test datasets:"
echo "  - CT-RATE: $CTRATE_TEST_CT (18 pathologies)"
echo ""

# Run testing
python run_test.py \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --ctrate_test_ct_path "$CTRATE_TEST_CT" \
    --ctrate_test_label_path "$CTRATE_TEST_LABELS" \
    --save_dir "$RESULTS_DIR" \
    --batch_size 8 \
    --num_workers 8 \
    --context_length 77 \
    --dinov2_model_name "dinov2_vitb14" \
    --test_ctrate \
    --save_predictions

echo ""
echo "Testing completed!"
echo "Results saved to: $RESULTS_DIR"
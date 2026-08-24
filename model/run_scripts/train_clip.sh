#!/bin/bash
#SBATCH --job-name=CLIP_3D
#SBATCH --output=clip_3d_optimized_%j.out
#SBATCH --error=clip_3d_optimized_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=160G
#SBATCH --partition=ai
#SBATCH --mail-type=END,FAIL

# Load CUDA module
module load cuda/12.4

# Load conda environment
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
REPO_PATH="${ACT_REPO:-/path/to/ACT}/model"

# Training datasets - multiple H5 and CSV files (INSPECT excluded)
TRAIN_CT_PATHS=(
    "/path/to/data/ctrate_train.h5"
    "/path/to/data/merlin_train.h5"
)
TRAIN_TXT_PATHS=(
    "${ACT_REPO:-/path/to/ACT}/model/data/ct_rate/train_reports.csv"
    "${ACT_REPO:-/path/to/ACT}/model/data/merlin/train_reports.csv"
)

# Validation and test paths (CT-RATE)
VAL_CT_PATH="/path/to/data/ctrate_valid.h5"
VAL_LABEL_PATH="${ACT_REPO:-/path/to/ACT}/model/data/ct_rate/valid_predicted_labels.csv"
TEST_CT_PATH="/path/to/data/ctrate_test.h5"
TEST_LABEL_PATH="${ACT_REPO:-/path/to/ACT}/model/data/ct_rate/test_predicted_labels.csv"

SAVE_DIR="/path/to/models/clip_3d/"

# Create save directory
mkdir -p $SAVE_DIR

# Change to repository directory
cd $REPO_PATH

# Run 3D CLIP training with DDP on multiple datasets
echo "Starting 3D CLIP training with DDP on 2 GPUs..."
echo "Training configuration:"
echo "  - Using OpenCLIP optimizations for memory efficiency"
echo "  - Loss: CLIP Optimized (with local loss and gather_with_grad)"
echo "  - Batch size: 1 per GPU x 2 GPUs = 2 per step"
echo "  - Gradient accumulation: 16 steps"
echo "  - Effective batch size: 2 * 16 = 32 samples per optimization step"
echo "  - Model: DinoV3 ViT-B/14 with Transformer Fusion"
echo "  - Warmup: 100 optimization steps (3,200 samples)"
echo "  - Validation: Every 50 optimization steps (1,600 samples)"
echo "  - Logging: Every 5 optimization steps (160 samples)"
echo ""
echo "Training datasets:"
echo "  - CT-RATE: ${TRAIN_CT_PATHS[0]}"
echo "  - MERLIN: ${TRAIN_CT_PATHS[1]}"
echo ""
echo "Validation datasets:"
echo "  - CT-RATE: $VAL_CT_PATH (18 pathologies)"
echo ""

# Set NCCL timeout to 30 minutes (1800 seconds)
export NCCL_TIMEOUT=1800

torchrun --nproc_per_node=2 run_train.py \
    --use_ddp \
    --ct_filepath "${TRAIN_CT_PATHS[@]}" \
    --txt_filepath "${TRAIN_TXT_PATHS[@]}" \
    --val_ct_filepath "$VAL_CT_PATH" \
    --val_label_path "$VAL_LABEL_PATH" \
    --test_ct_filepath "$TEST_CT_PATH" \
    --test_label_path "$TEST_LABEL_PATH" \
    --save_dir "$SAVE_DIR" \
    --batch_size 1 \
    --epochs 40 \
    --lr 1e-4 \
    --weight_decay 0.2 \
    --warmup_steps 500 \
    --accum_freq 16 \
    --dinov2_model_name "dinov2_vitl14" \
    --dino_version "v3" \
    --fusion_method "transformer" \
    --fusion_depth 4 \
    --context_length 77 \
    --loss_type "clip_optimized" \
    --local_loss \
    --gather_with_grad \
    --cache_labels \
    --do_validate \
    --valid_interval 200 \
    --val_batch_size 8 \
    --test_batch_size 4 \
    --log_interval 5 \
    --model_name "clip_3d_grad_accu_vitL" \
    --column "Impressions_EN" "Impressions_EN" \
    --seed 42 \
    --test_after_training \
    --num_workers 8

echo "Training completed!"
echo "Model saved to: $SAVE_DIR"
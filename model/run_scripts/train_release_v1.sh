#!/bin/bash
#SBATCH --job-name=ACT_Train_Release_V1
#SBATCH --output=act_train_release_v1_%j.out
#SBATCH --error=act_train_release_v1_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=160G
#SBATCH --partition=ai
#SBATCH --mail-type=END,FAIL

# ============================================================================
# Training recipe for the released checkpoint: clip_3d_ctrate_merlin_v1
# ============================================================================
# The released best_model.pt is a bare state_dict (no config is stored in the
# checkpoint), so this launcher was RECONSTRUCTED from the original SLURM
# launcher preserved in the training repo's git history (run_scripts/
# train_clip_3d.sh, July 2025) and verified against the checkpoint's tensor
# shapes and the run's validation_log.csv.
#
# Recovered configuration (all verified against the released state_dict):
#   - Visual backbone: DINOv2 ViT-B/14 with registers (torch.hub
#     "dinov2_vitb14" + "_reg" suffix appended by train.py; 12 blocks,
#     dim 768, 4 register tokens)
#   - Slice fusion:    x_transformers Encoder, depth 2, learned CLS token,
#     rotary position embeddings, followed by a Linear 768->768 projection.
#     NOTE: the training-time Encoder used heads=12 and ff_mult=2, and the
#     train.py in this release builds exactly that configuration, so the
#     released code loads clip_3d_ctrate_merlin_v1 as shipped.
#   - Text encoder:    CLIP text transformer (width 512, 12 layers, 8 heads,
#     context length 77, vocab 49408), projected to the shared 768-d space
#   - Loss:            symmetric InfoNCE (CLIP CrossEntropy loss), computed
#     per GPU on the local 4x4 similarity matrix with NO cross-device
#     feature gathering (the July 2025 training code had no all_gather;
#     gradients were synchronized by DDP only)
#   - Optimizer:       AdamW, lr 1e-4, weight decay 0.2, cosine schedule
#     with 500 warmup steps
#   - Batching:        batch_size 4 per GPU x 2 A100s, gradient accumulation
#     32 micro-batches per optimizer step (the 2025 code summed the plain
#     gradients of 32 independent per-GPU 4-sample InfoNCE losses; the
#     contrastive matrix stayed 4x4). See caveat below.
#   - Data:            CT-RATE train + Merlin (all splits combined), text
#     column "Impressions_EN" in both report CSVs; 160x224x224 uint8 HDF5
#     volumes from preprocessing/run_preprocess.py
#   - Seed 42; validation on CT-RATE every 200 steps (18 pathologies,
#     val batch 4); no early stopping. The run was stopped during epoch 11;
#     best_model.pt is the checkpoint with the highest CT-RATE validation
#     mean AUC (0.7560, step 4000 of epoch 2 in validation_log.csv).
#
# CAVEAT on exact reproduction: the current run_train.py implements
# gradient accumulation with OpenCLIP's two-pass scheme (--accum_freq),
# which computes the contrastive loss over the full accumulated batch
# (128 samples per GPU here) instead of summing 32 independent 4-sample
# losses, and its DDP CLIP path gathers text features across GPUs. Both
# differ from the July 2025 objective described above. This launcher passes
# the recovered values through the current flags; for a bit-faithful rerun,
# use the training repo at the July 2025 commit.
# ============================================================================

# Load CUDA module
module load cuda/12.4

# Load conda environment
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate ctproject

# Set paths
REPO_PATH="${ACT_REPO:-/path/to/ACT}/model"

# Training datasets (INSPECT excluded from training)
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

torchrun --nproc_per_node=2 run_train.py \
    --use_ddp \
    --ct_filepath "${TRAIN_CT_PATHS[@]}" \
    --txt_filepath "${TRAIN_TXT_PATHS[@]}" \
    --val_ct_filepath "$VAL_CT_PATH" \
    --val_label_path "$VAL_LABEL_PATH" \
    --test_ct_filepath "$TEST_CT_PATH" \
    --test_label_path "$TEST_LABEL_PATH" \
    --save_dir "$SAVE_DIR" \
    --batch_size 4 \
    --epochs 40 \
    --lr 1e-4 \
    --weight_decay 0.2 \
    --warmup_steps 500 \
    --accum_freq 32 \
    --dinov2_model_name "dinov2_vitb14" \
    --dino_version "v2" \
    --fusion_method "transformer" \
    --fusion_depth 2 \
    --context_length 77 \
    --loss_type "clip" \
    --do_validate \
    --valid_interval 200 \
    --val_batch_size 4 \
    --test_batch_size 2 \
    --log_interval 10 \
    --model_name "clip_3d_ctrate_merlin_v1" \
    --column "Impressions_EN" "Impressions_EN" \
    --seed 42 \
    --test_after_training \
    --num_workers 8

echo "Training completed!"
echo "Model saved to: $SAVE_DIR"

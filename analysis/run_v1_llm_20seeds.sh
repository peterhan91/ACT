#!/bin/bash
# 20-seed run for v1 / linear_<LLM> across all three labelsets at each
# labelset's best LR (read from bestlr.txt files written by the LR sweeps).
# Concept-importance audits are produced per seed by exp_phenotype_ct.py.
#
# Usage:  bash run_v1_llm_20seeds.sh <LLM>          where LLM ∈ {sfr,harrier,openai}
set -e

LLM="${1:?missing LLM (sfr|harrier|openai)}"

export CLIP3D_RUN_TAG=v1
export CLIP3D_BEST=clip_3d_ctrate_merlin_v1
export PY=python

cd ${ACT_REPO:-/path/to/ACT}/analysis

LOG_DIR="logs/v1/seeds_${LLM}"
mkdir -p "$LOG_DIR"

CTRATE18_BESTLR=$(cat "outputs/v1/cbm/linear_${LLM}__bestlr.txt" 2>/dev/null || echo "")
PE_BESTLR=$(cat "outputs/v1/external/pe_pheno__linear_${LLM}__bestlr.txt" 2>/dev/null || echo "")
PHENO_BESTLR=$(cat "outputs/v1/external/phenotype__linear_${LLM}__bestlr.txt" 2>/dev/null || echo "")

if [[ -z "$CTRATE18_BESTLR" || -z "$PE_BESTLR" || -z "$PHENO_BESTLR" ]]; then
  echo "[20seeds/${LLM}] ERROR: missing bestlr.txt — run the LR sweeps first." >&2
  echo "  CT-RATE-18: '$CTRATE18_BESTLR'" >&2
  echo "  PE-3:       '$PE_BESTLR'" >&2
  echo "  phenotype:  '$PHENO_BESTLR'" >&2
  exit 1
fi
echo "[20seeds/${LLM}] CT-RATE-18 best LR = $CTRATE18_BESTLR"
echo "[20seeds/${LLM}] PE-3        best LR = $PE_BESTLR"
echo "[20seeds/${LLM}] phenotype   best LR = $PHENO_BESTLR"

SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

run_ctrate18 () {
  local SEED=$1 LR=$2
  local SUF="__seed${SEED}__lr${LR}"
  echo "=== [ctrate18/${LLM} seed=$SEED lr=$LR] $(date) ==="
  $PY exp_linear_ct.py \
      --llms "$LLM" \
      --lr "$LR" \
      --seed "$SEED" \
      --out_suffix "$SUF" \
      2>&1 | tee "$LOG_DIR/ctrate18_seed${SEED}.log"
}

run_pe () {
  local SEED=$1 LR=$2
  local SUF="__seed${SEED}__lr${LR}"
  echo "=== [pe/${LLM} seed=$SEED lr=$LR] $(date) ==="
  $PY exp_phenotype_ct.py \
      --labelsets pe \
      --methods "linear_${LLM}" \
      --lr "$LR" \
      --seed "$SEED" \
      --out_suffix "$SUF" \
      2>&1 | tee "$LOG_DIR/pe_seed${SEED}.log"
}

run_phenotype () {
  local SEED=$1 LR=$2
  local SUF="__seed${SEED}__lr${LR}"
  echo "=== [phenotype/${LLM} seed=$SEED lr=$LR] $(date) ==="
  $PY exp_phenotype_ct.py \
      --labelsets phenotype \
      --methods "linear_${LLM}" \
      --lr "$LR" \
      --seed "$SEED" \
      --out_suffix "$SUF" \
      2>&1 | tee "$LOG_DIR/phenotype_seed${SEED}.log"
}

echo "[20seeds/${LLM}] start  $(date)"
for S in "${SEEDS[@]}"; do
  run_ctrate18  "$S" "$CTRATE18_BESTLR"
  run_pe        "$S" "$PE_BESTLR"
  run_phenotype "$S" "$PHENO_BESTLR"
done
echo "[20seeds/${LLM}] all training done  $(date)"

echo "[20seeds/${LLM}] aggregating..."
$PY aggregate_20seeds.py \
    --llm             "$LLM" \
    --ctrate18_bestlr "$CTRATE18_BESTLR" \
    --pe_bestlr       "$PE_BESTLR" \
    --pheno_bestlr    "$PHENO_BESTLR" \
    2>&1 | tee "$LOG_DIR/aggregate.log"

echo "[20seeds/${LLM}] done  $(date)"

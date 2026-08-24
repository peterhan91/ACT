#!/bin/bash
# Generic full-pipeline runner for any 3D-CLIP checkpoint.
# Usage: bash run_pipeline.sh <run_tag> <clip3d_best_name>
#   <run_tag>           short label used for outputs/<tag>/ + concept_bank.clip_text_emb.<tag>.npz
#   <clip3d_best_name>  config name in clip_3d_eval/configs.json
set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <run_tag> <clip3d_best_name>"
    exit 1
fi

export CLIP3D_RUN_TAG=$1
export CLIP3D_BEST=$2
export PY=python
cd ${ACT_REPO:-/path/to/ACT}/analysis

LOG=logs/${CLIP3D_RUN_TAG}
mkdir -p "$LOG"

echo "=== [${CLIP3D_RUN_TAG}] config: ${CLIP3D_BEST} ==="
date

# 0. Build clip_text concept bank for this checkpoint (~75s).
if [ ! -f concept_bank.clip_text_emb.${CLIP3D_RUN_TAG}.npz ]; then
    echo "=== [${CLIP3D_RUN_TAG}] 0. clip_text concept bank ==="
    date
    $PY get_embed_ct.py --skip_sfr --skip_harrier --skip_openai --clip_bs 128 \
        > "$LOG/00_clip_text_bank.log" 2>&1
fi

# 1. zeroshot CT-RATE test (plain + sfr + harrier + openai if available)
echo "=== [${CLIP3D_RUN_TAG}] 1. zeroshot CT-RATE ==="
date
# openai is included when (a) the concept bank exists AND (b) the precomputed
# 18-label cache is already present. The runtime API call is blocked on GH200,
# so we never attempt it from here — see precompute_openai_labels.py.
METHODS="plain sfr harrier"
if [ -f concept_bank.openai_emb.npz ] && \
   [ -f outputs/${CLIP3D_RUN_TAG}/cache/label_emb.openai.npz ]; then
    METHODS="$METHODS openai"
fi
$PY exp_zeroshot_ct.py --datasets ctrate_test --methods $METHODS \
    > "$LOG/01_zeroshot.log" 2>&1

# 2. CBM CT-RATE (encodes ctrate_train img_feats; ~30 min)
echo "=== [${CLIP3D_RUN_TAG}] 2. CBM ==="
date
$PY exp_cbm_ct.py --batch_size 512 --encode_batch 4 --epochs 200 --patience 10 \
    > "$LOG/02_cbm.log" 2>&1

# 3. Linear probes on llm_repr
echo "=== [${CLIP3D_RUN_TAG}] 3. linear probes ==="
date
LLMS="sfr harrier"
if [ -f concept_bank.openai_emb.npz ] && \
   [ -f outputs/${CLIP3D_RUN_TAG}/cache/label_emb.openai.npz ]; then
    LLMS="$LLMS openai"
fi
$PY exp_linear_ct.py --llms $LLMS --batch_size 512 --epochs 200 --patience 10 \
    > "$LOG/03_linear.log" 2>&1

# 4. Audit (cosine alignment for each LLM probe)
echo "=== [${CLIP3D_RUN_TAG}] 4. audit ==="
date
for llm in $LLMS; do
    coefs="outputs/${CLIP3D_RUN_TAG}/cbm/linear_${llm}__coefs.pt"
    if [ -f "$coefs" ]; then
        $PY concept_audit_ct.py --llm_coefs "$coefs" --llm "$llm" \
            --out_suffix ".linear_${llm}" --top_k_per_label 100 \
            > "$LOG/04_audit_${llm}.log" 2>&1
    fi
done

# 5. External INSPECT (encodes inspect_* img_feats; ~60 min)
echo "=== [${CLIP3D_RUN_TAG}] 5. external INSPECT ==="
date
$PY run_external_ct.py --datasets inspect_valid inspect_test inspect_train \
    --labelsets predicted pe --methods $METHODS \
    --batch_size 4 --num_workers 4 \
    > "$LOG/05_external.log" 2>&1

# 6. Phenotype eval (manifest splits, 221 phecodes filter, openai if available)
echo "=== [${CLIP3D_RUN_TAG}] 6. phenotype eval ==="
date
PH_METHODS="plain sfr harrier linear_sfr linear_harrier cbm"
if [ -f concept_bank.openai_emb.npz ] && \
   [ -f outputs/${CLIP3D_RUN_TAG}/cache/label_emb.openai.npz ]; then
    PH_METHODS="$PH_METHODS openai linear_openai"
fi
$PY exp_phenotype_ct.py --min_test_positives 50 --epochs 200 --patience 10 \
    --methods $PH_METHODS \
    > "$LOG/06_phenotype.log" 2>&1

echo "=== [${CLIP3D_RUN_TAG}] DONE ==="
date

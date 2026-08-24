#!/bin/bash
# C=1e6 (unregularized) LBFGS trusted-concept probe + audit, WITH train bootstrap.
# Faithful re-run of the OLD purification setting (run_purification_eval_f2llm.sh:
# trusted_concept_probe.py --solver lbfgs --lbfgs_C 1e6 + _audit_trusted_openai_probes.py)
# with the ONLY change being a 20-seed TRAIN bootstrap around each deterministic fit.
# Reports per (label, method): AUROC point + bootstrap mean/std/CI, and per-concept
# signed importance point + bootstrap mean/std + sign-stability.
# CPU/many-core bound: joblib n_jobs=-1 saturates the 72 Grace cores (single-thread BLAS).

PY=python
cd ${ACT_REPO:-/path/to/ACT}/analysis || exit 1
export CLIP3D_RUN_TAG=v1
LLM=f2llm
LOG=logs/v1/purification_${LLM}
mkdir -p "$LOG"

# methods: default trusted+direct (purified subspaces). Pass "original_topk trusted direct"
# as args to also run the heavy 221-label without-purification baseline (~+1.5h).
METHODS="${@:-trusted direct}"
echo "=== [trusted bootstrap ${LLM}] START $(date) | methods=$METHODS ==="
$PY _audit_trusted_bootstrap.py \
    --llm "$LLM" \
    --methods $METHODS \
    --C 1e6 --max_iter 5000 \
    --n_seeds 20 \
    --original_top_k 100 \
    --n_jobs -1
echo "=== [trusted bootstrap ${LLM}] DONE exit=$? $(date) ==="

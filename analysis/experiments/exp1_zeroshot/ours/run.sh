#!/bin/bash
# CLEAR_CT_EXPS / exp1_zeroshot — OUR model.
# Zero-shot CT finding classification with the canonical v1 3D-CLIP checkpoint
# clip_3d_ctrate_merlin_v1 (dinov2 vitb14, transformer fusion depth 2), reusing
# the clip_3d_concepts clear3d package + its cached v1 features. Plain + OpenAI
# modes on CT-RATE (test), RAD-ChestCT, RSNA-2023 trauma (test).
#
# Usage:
#   ./run.sh                         # all 3 datasets, plain + openai
#   ./run.sh --datasets rsna2023_test
#   ./run.sh --modes plain
set -euo pipefail

# --- our model selection (canonical orchestrator pattern, see repo run_*.sh) ---
export CLIP3D_RUN_TAG=v1
export CLIP3D_BEST=clip_3d_ctrate_merlin_v1
# Repo that provides the clear3d package + cached features/outputs.
export CLEAR3D_REPO=${ACT_REPO:-/path/to/ACT}/analysis

# Our (micromamba/conda) env — Python 3.12 aarch64, torch 2.11+cu130.
PY=${PY:-python}

# Pick up OPENAI_API_KEY from the repo .env if present (not needed when the
# 18-label / trauma OpenAI label embeddings are already cached, which they are).
for _envf in "$CLEAR3D_REPO/../.env" "$CLEAR3D_REPO/.env"; do [[ -f "$_envf" ]] && { set -a; source "$_envf"; set +a; break; }; done

cd "$(dirname "$0")"
echo "=== [ours] exp1 zero-shot | $CLIP3D_BEST ==="
date
"$PY" run_zeroshot.py "$@"
echo "=== [ours] DONE ==="
date

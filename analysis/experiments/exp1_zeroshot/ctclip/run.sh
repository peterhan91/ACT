#!/bin/bash
# CT-CLIP_v2 — exp1 zero-shot baseline. Native (own nii_img_to_tensor,
# 0.75/0.75/1.5mm->240x480x480) is the DEFAULT; --source h5 = controlled ablation.
# Chest model sharing our 18 labels -> all 3 datasets.
#   ./run.sh                                  # native, all 3 datasets
#   ./run.sh --source h5                      # controlled identical-input ablation
#   ./run.sh --datasets ctrate_test --limit 2 # smoke
set -euo pipefail
export CLEAR3D_REPO=${ACT_REPO:-/path/to/ACT}/analysis
PY=${PY:-python}
cd "$(dirname "$0")"
echo "=== [ctclip] exp1 zero-shot | CT-CLIP_v2 ==="
date
"$PY" run_zeroshot.py "$@"
echo "=== [ctclip] DONE ==="
date

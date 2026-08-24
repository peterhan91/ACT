#!/bin/bash
# M3D-CLIP (GoodBaiBai88/M3D-CLIP) — exp1 zero-shot baseline. Native (own
# CropForeground+min-max->32x256x256) is the DEFAULT; --source h5 = controlled ablation.
# General 3D CT CLIP -> all 3 datasets. Reuses the ctclip env (transformers+monai1.3).
#   ./run.sh                                   # native, all 3 datasets
#   ./run.sh --source h5                       # controlled identical-input ablation
#   ./run.sh --datasets ctrate_test --limit 4  # smoke
set -euo pipefail
export CLEAR3D_REPO=${ACT_REPO:-/path/to/ACT}/analysis
PY=${PY:-python}
cd "$(dirname "$0")"
echo "=== [m3dclip] exp1 zero-shot | GoodBaiBai88/M3D-CLIP ==="
date
"$PY" run_zeroshot.py "$@"
echo "=== [m3dclip] DONE ==="
date

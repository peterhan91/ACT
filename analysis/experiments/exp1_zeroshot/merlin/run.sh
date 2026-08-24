#!/bin/bash
# Merlin (Stanford) — exp1 zero-shot baseline. Native (own MONAI ImageTransforms,
# RAS/1.5x1.5x3/224x224x160) is the DEFAULT; --source h5 = controlled ablation.
# Abdominal model -> rsna2023_test is the native fit; chest sets are OOD reference.
#   ./run.sh                                   # native, all 3 datasets
#   ./run.sh --source h5                       # controlled identical-input ablation
#   ./run.sh --datasets rsna2023_test --limit 4   # smoke
set -euo pipefail
export CLEAR3D_REPO=${ACT_REPO:-/path/to/ACT}/analysis
PY=${PY:-python}
cd "$(dirname "$0")"
echo "=== [merlin] exp1 zero-shot | stanfordmimi/Merlin ==="
date
"$PY" run_zeroshot.py "$@"
echo "=== [merlin] DONE ==="
date

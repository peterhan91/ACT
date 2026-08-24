#!/bin/bash
# CT-CLIP native on radchest + rsna (extends its ctrate-only native via the unified
# .npz/NIfTI -> (HU,affine) -> temp-NIfTI loader). CT-CLIP is its own model/summary so
# no race with the COLIPRI/Merlin radchest run. Image features are cached. Re-aggregates.
#   nohup ./run_ctclip_extra.sh > run_ctclip_extra.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"
export CLEAR3D_REPO=${ACT_REPO:-/path/to/ACT}/analysis
echo "===== [$(date +%F\ %H:%M:%S)] CT-CLIP native : radchest + rsna ====="
python ctclip/run_zeroshot.py \
    --source native --datasets radchest rsna2023_test --batch_size 2 --num_workers 4
echo "===== [$(date +%F\ %H:%M:%S)] aggregating ====="
python aggregate_results.py
echo "===== [$(date +%F\ %H:%M:%S)] DONE ====="

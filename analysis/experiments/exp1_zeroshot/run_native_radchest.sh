#!/bin/bash
# Native RAD-ChestCT (raw .npz, 0.8mm iso HU, LPS — orientation verified).
# Runs the Merlin native radchest evaluation, then re-aggregates.
# CT-CLIP stays ctrate-only. The COLIPRI baseline was retired from the release
# (its runner was archived and is not part of the paper's benchmark).
# Detached:  nohup ./run_native_radchest.sh > run_native_radchest.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"
export CLEAR3D_REPO=${ACT_REPO:-/path/to/ACT}/analysis
log(){ echo; echo "===== [$(date +%F\ %H:%M:%S)] $* ====="; }

log "Merlin native radchest (MONAI RAS/1.5x1.5x3, 224x224x160) : 3630 vols"
python merlin/run_zeroshot.py \
    --source native --datasets radchest --batch_size 4 --num_workers 8

log "aggregate native + h5 results"
python aggregate_results.py
log "DONE"

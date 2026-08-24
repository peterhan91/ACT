#!/bin/bash
# Full f-VLM zero-shot eval. Waits for mask generation (gen_masks.py) to finish, then
# runs on all of ctrate_test (organ-ROI forward_test_win over the 16 organ-tied
# pathologies), then re-aggregates. Detached:
#   nohup ./run_fvlm.sh > run_fvlm.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"                                   # exp1_zeroshot/fvlm
export CLEAR3D_REPO=${ACT_REPO:-/path/to/ACT}/analysis
if [ -f gen_masks.pid ]; then
  echo "===== [$(date +%F\ %H:%M:%S)] waiting for mask-gen (PID $(cat gen_masks.pid)) ====="
  while kill -0 "$(cat gen_masks.pid)" 2>/dev/null; do sleep 60; done
fi
echo "===== [$(date +%F\ %H:%M:%S)] masks=$(ls masks/*.nii.gz 2>/dev/null | wc -l); running full f-VLM eval ====="
python run_zeroshot.py --datasets ctrate_test
echo "===== [$(date +%F\ %H:%M:%S)] aggregating ====="
python ../aggregate_results.py
echo "===== [$(date +%F\ %H:%M:%S)] DONE ====="

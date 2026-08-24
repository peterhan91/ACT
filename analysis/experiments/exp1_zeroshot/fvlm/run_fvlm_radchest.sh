#!/bin/bash
# f-VLM on RAD-ChestCT. Waits for the ctrate f-VLM run to finish (sequences the two
# GPU-heavy TotalSegmentator passes + the two f-VLM evals), then generates radchest
# 4-organ masks (3630), runs f-VLM radchest eval, re-aggregates. Detached:
#   nohup ./run_fvlm_radchest.sh > run_fvlm_radchest.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"                                   # exp1_zeroshot/fvlm
export CLEAR3D_REPO=${ACT_REPO:-/path/to/ACT}/analysis
if [ -f run_fvlm.pid ]; then
  echo "===== [$(date +%F\ %H:%M:%S)] waiting for ctrate f-VLM run (PID $(cat run_fvlm.pid)) ====="
  while kill -0 "$(cat run_fvlm.pid)" 2>/dev/null; do sleep 60; done
fi
echo "===== [$(date +%F\ %H:%M:%S)] generating radchest 4-organ masks (3630) ====="
python gen_masks.py --dataset radchest
echo "===== [$(date +%F\ %H:%M:%S)] f-VLM radchest eval ====="
python run_zeroshot.py --datasets radchest
echo "===== [$(date +%F\ %H:%M:%S)] aggregating ====="
python ../aggregate_results.py
echo "===== [$(date +%F\ %H:%M:%S)] DONE ====="

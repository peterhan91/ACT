#!/bin/bash
# medsiglip on the SCALED PMBB pools (bf16, slice_bs=64). The cost is the 448px
# SigLIP vision tower over every axial slice; runs on this GH200 (shared w/ giannoua).
# Re-aggregates at the end so RESULTS_PMBB_EXP1_BIG gains the medsiglip row.
# Detached: nohup ./run_pmbb_medsiglip.sh > _logs_pmbb_medsiglip/run.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"
LOG="_logs_pmbb_medsiglip"; mkdir -p "$LOG"
ML=python
echo "==== medsiglip scaled QUEUED @ $(date) on $(hostname) (pid $$) ===="
# Run after the heavy native runs (merlin, ctclip) have freed the GPU —
# medsiglip's 448px tower needs ~10GB.
echo "==== medsiglip scaled START @ $(date) ===="
"$ML" run_zeroshot_2d.py --model medsiglip --datasets pmbb_chest_nc pmbb_abd_ce \
  && echo "[medsiglip] OK" || echo "[medsiglip] FAIL"
echo "==== aggregating @ $(date) ===="
python aggregate_pmbb_exp1_big.py > "$LOG/aggregate.log" 2>&1 \
  && echo "[agg] OK" || echo "[agg] FAIL (see $LOG/aggregate.log)"
echo "==== medsiglip scaled DONE @ $(date) ===="

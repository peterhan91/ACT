#!/bin/bash
# exp1_zeroshot — PMBB chest/abd zero-shot finding classification, all models.
# Labels are phrase-mined (pmbb_labels/). Two parallel streams, each SEQUENTIAL
# internally (no GPU OOM):
#   3D stream: ours(plain+openai) -> colipri -> merlin -> m3dclip -> ctclip   (native; ctclip heaviest last)
#   2D stream: openai_clip -> biomedclip -> medsiglip                          (h5 + own per-slice pipeline)
# Each writes pmbb_* rows to its <model>/results/summary.csv, then aggregate.
#
# Detached:  nohup ./run_pmbb_exp1.sh > _logs_pmbb_exp1/run.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"
LOG="_logs_pmbb_exp1"; mkdir -p "$LOG"
DS="pmbb_chest_test pmbb_abd_test"
echo "==== run_pmbb_exp1 START @ $(date) (pid $$) | datasets=[$DS] ===="

# ---- 3D stream (native, heavy) ----
(
  echo "[3D] start @ $(date)"
  echo "[3D] ours @ $(date)";    ./ours/run.sh    --datasets $DS --modes plain openai && echo "[3D] ours OK"    || echo "[3D] ours FAIL"
# (COLIPRI baseline retired from the release; its runner was archived)
  echo "[3D] merlin @ $(date)";  ./merlin/run.sh  --datasets $DS --source native       && echo "[3D] merlin OK"  || echo "[3D] merlin FAIL"
  echo "[3D] m3d @ $(date)";     ./m3dclip/run.sh --datasets $DS --source native       && echo "[3D] m3d OK"     || echo "[3D] m3d FAIL"
  echo "[3D] ctclip @ $(date)";  ./ctclip/run.sh  --datasets $DS --source native       && echo "[3D] ctclip OK"  || echo "[3D] ctclip FAIL"
  echo "[3D] DONE @ $(date)"
) > "$LOG/stream_3d.log" 2>&1 &
P3D=$!

# ---- 2D stream (h5 + own per-slice pipeline) ----
ML=python
(
  echo "[2D] start @ $(date)"
  for m in openai_clip biomedclip medsiglip; do
    echo "[2D] $m @ $(date)"
    "$ML" run_zeroshot_2d.py --model "$m" --datasets $DS && echo "[2D] $m OK" || echo "[2D] $m FAIL"
  done
  echo "[2D] DONE @ $(date)"
) > "$LOG/stream_2d.log" 2>&1 &
P2D=$!

wait $P3D $P2D
echo "==== both streams done @ $(date); aggregating ===="
python aggregate_pmbb_exp1.py > "$LOG/aggregate.log" 2>&1 \
  && echo "[agg] OK" || echo "[agg] FAIL (see $LOG/aggregate.log)"
echo "==== run_pmbb_exp1 FULLY DONE @ $(date) ===="

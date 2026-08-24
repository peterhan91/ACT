#!/bin/bash
# exp1_zeroshot — SCALED PMBB zero-shot finding classification (supersedes the
# 2,489 pilot). Pools: pmbb_chest_nc (non-contrast chest, 9,097) + pmbb_abd_ce
# (contrast/Merlin-style abd, 14,290). COLIPRI DROPPED (was ~79% of cost).
# Runs on the GH200 (gracehopper1). Three concurrent streams to use the GPU fully:
#   A) 3D native baselines, SEQUENTIAL: merlin -> m3dclip -> ctclip  (I/O-bound, many workers)
#   B) ours (crimson, h5, GPU-bound)                                  (concurrent w/ A & C)
#   C) 2D baselines, SEQUENTIAL: openai_clip -> biomedclip -> medsiglip
# Each writes pmbb_{chest_nc,abd_ce} rows to its <model>/results/summary.csv.
#
# Detached:  nohup ./run_pmbb_exp1_big.sh > _logs_pmbb_exp1_big/run.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"
LOG="_logs_pmbb_exp1_big"; mkdir -p "$LOG"
DS="pmbb_chest_nc pmbb_abd_ce"
NW=24   # DataLoader workers for the I/O-bound native baselines (72 cores total)
echo "==== run_pmbb_exp1_big START @ $(date) on $(hostname) (pid $$) | datasets=[$DS] ===="

# ---- Stream A: 3D native baselines (sequential; ctclip heaviest last) ----
(
  echo "[A] start @ $(date)"
  echo "[A] merlin @ $(date)";  ./merlin/run.sh  --datasets $DS --source native --batch_size 8 --num_workers $NW && echo "[A] merlin OK"  || echo "[A] merlin FAIL"
  echo "[A] m3d @ $(date)";     ./m3dclip/run.sh --datasets $DS --source native --batch_size 8 --num_workers $NW && echo "[A] m3d OK"     || echo "[A] m3d FAIL"
  echo "[A] ctclip @ $(date)";  ./ctclip/run.sh  --datasets $DS --source native --batch_size 4 --num_workers $NW && echo "[A] ctclip OK"  || echo "[A] ctclip FAIL"
  echo "[A] DONE @ $(date)"
) > "$LOG/stream_A_native3d.log" 2>&1 &
PA=$!

# ---- Stream B: ours (h5; fresh encode) ----
(
  echo "[B] start @ $(date)"
  echo "[B] ours @ $(date)"; ./ours/run.sh --datasets $DS --modes plain --batch_size 8 --num_workers 8 && echo "[B] ours OK" || echo "[B] ours FAIL"
  echo "[B] DONE @ $(date)"
) > "$LOG/stream_B_ours.log" 2>&1 &
PB=$!

# ---- Stream C: 2D baselines (sequential) ----
ML=python
(
  echo "[C] start @ $(date)"
  for m in openai_clip biomedclip medsiglip; do
    echo "[C] $m @ $(date)"
    "$ML" run_zeroshot_2d.py --model "$m" --datasets $DS && echo "[C] $m OK" || echo "[C] $m FAIL"
  done
  echo "[C] DONE @ $(date)"
) > "$LOG/stream_C_2d.log" 2>&1 &
PC=$!

wait $PA $PB $PC
echo "==== all streams done @ $(date); aggregating ===="
python aggregate_pmbb_exp1_big.py > "$LOG/aggregate.log" 2>&1 \
  && echo "[agg] OK" || echo "[agg] FAIL (see $LOG/aggregate.log)"
echo "==== run_pmbb_exp1_big FULLY DONE @ $(date) ===="

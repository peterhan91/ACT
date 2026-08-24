#!/bin/bash
# QUEUED f-VLM mask generation for pmbb_chest_nc (the one missing chest dataset).
# Waits until the 2D native batch finishes (frees the GH200), then runs
# TotalSegmentator (fast) sharded 4 ways over the 9,097 volumes -> masks_pmbb_chest_nc/
# keyed by VolumeName. Resumable (skips existing). TotalSegmentator lives in the
# totalseg_venv env; the f-VLM zero-shot (separate, after masks) uses the fvlm env.
#
# Detached:  nohup ./run_fvlm_pmbb_masks.sh > _logs_pmbb_masks/run_all.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"                                   # fvlm/
TSPY=python
NSH=12
export OMP_NUM_THREADS=6
LOG=_logs_pmbb_masks; mkdir -p "$LOG"

echo "[queue] waiting for the 2D native batch to finish (frees GPU) @ $(date)"
until ! pgrep -f 'run_zeroshot_2d[.]py' >/dev/null 2>&1; do sleep 120; done
echo "[queue] 2D batch done; starting pmbb_chest_nc masks, $NSH shards @ $(date)"

for s in $(seq 0 $((NSH - 1))); do
  ( echo "[shard $s/$NSH] start @ $(date)"
    "$TSPY" gen_masks.py --dataset pmbb_chest_nc --nshards "$NSH" --shard "$s"
    echo "[shard $s/$NSH] done @ $(date)" ) > "$LOG/shard${s}.log" 2>&1 &
done
wait
echo "[queue] pmbb_chest_nc masks DONE @ $(date) (have $(ls masks_pmbb_chest_nc 2>/dev/null | wc -l)/9097)"
echo "next: $TSPY ... (run f-VLM zero-shot via fvlm env on pmbb_chest_nc)"

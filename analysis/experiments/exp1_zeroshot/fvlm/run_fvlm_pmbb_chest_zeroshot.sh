#!/bin/bash
# CHAINED f-VLM zero-shot on pmbb_chest_nc (the only PMBB set f-VLM applies to —
# it's a chest-organ model, so the abdominal sets rsna2023_test / pmbb_abd_ce are
# out of scope). Waits for the queued TotalSegmentator mask gen
# (run_fvlm_pmbb_masks.sh -> "masks DONE" in _logs_pmbb_masks/run_all.log) to finish,
# then runs the f-VLM zero-shot SHARDED 4 ways over the ~9,097 chest volumes on the
# now-free GH200 (each shard = a contiguous index block, own model build), and
# finalizes: concat shard probs -> score -> fvlm/results/pmbb_chest_nc__native__*.
#
# Detached:  nohup ./run_fvlm_pmbb_chest_zeroshot.sh > _logs_pmbb_zeroshot/run_all.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"                                   # fvlm/
FVLMPY=python
DS=pmbb_chest_nc
NSH="${FVLM_NSHARDS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"         # cap per-proc CPU threads (NSH procs)
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
LOG=_logs_pmbb_zeroshot; mkdir -p "$LOG"

echo "==== f-VLM pmbb_chest_nc zero-shot START @ $(date) | shards=$NSH omp=$OMP_NUM_THREADS ===="
echo "[queue] waiting for pmbb_chest_nc mask gen to finish (masks DONE marker) @ $(date)"
until grep -q "masks DONE" _logs_pmbb_masks/run_all.log 2>/dev/null; do sleep 120; done
NMASK=$(ls "masks_${DS}" 2>/dev/null | wc -l)
echo "[queue] masks done (have $NMASK in masks_${DS}); starting zero-shot @ $(date)"

for s in $(seq 0 $((NSH - 1))); do
  ( echo "[shard $s/$NSH] start @ $(date)"
    "$FVLMPY" run_zeroshot.py --datasets "$DS" --nshards "$NSH" --shard "$s"
    echo "[shard $s/$NSH] done @ $(date)" ) > "$LOG/shard${s}.log" 2>&1 &
done
wait
echo "[finalize] concat shard probs + score @ $(date)"
"$FVLMPY" run_zeroshot.py --datasets "$DS" --nshards "$NSH" --finalize > "$LOG/finalize.log" 2>&1
echo "==== f-VLM pmbb_chest_nc zero-shot DONE @ $(date) ===="
grep -hE "mean AUC|Summary" "$LOG/finalize.log" 2>/dev/null || true
echo "re-aggregate big PMBB table:  cd .. && python aggregate_pmbb_exp1_big.py"

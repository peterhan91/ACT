#!/bin/bash
# 2D-only RECOVERY for inspect_pheno (3D natives + ours already done on disk).
# Fixed loader (native.load_hu_affine, robust to 4D/degenerate affines). GPU is free
# now, so 8 shards/model run concurrently. Verifies every shard saved before finalize
# (so a shard failure can't masquerade as a false "OK").
#   nohup ./run_inspect_pheno_2d.sh >> _logs_inspect_pheno/run.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"
export CLEAR3D_REPO=${ACT_REPO:-/path/to/ACT}/analysis
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
LOG=_logs_inspect_pheno; mkdir -p "$LOG"
ML=python
CRIMSON=python
DS=inspect_pheno
CACHE=../_cache2d

echo "######## 2D RECOVERY START $(date) (3D+ours already on disk) ########"

run2d () {  # model nshards
  local model="$1" N="$2" i missing=0
  for i in $(seq 0 $((N-1))); do
    "$ML" run_zeroshot_2d.py --model "$model" --datasets $DS --nshards "$N" --shard "$i" \
        > "$LOG/${model}_sh${i}.log" 2>&1 &
    sleep 1
  done
  wait
  for i in $(seq 0 $((N-1))); do
    [ -f "$CACHE/${model}__inspect_pheno__shard${i}of${N}.npy" ] || { echo "[$model] MISSING shard ${i}/${N}"; missing=1; }
  done
  if [ "$missing" -eq 0 ]; then
    "$ML" run_zeroshot_2d.py --model "$model" --datasets $DS --nshards "$N" --finalize \
        > "$LOG/${model}_final.log" 2>&1 && echo "[$model] OK" || echo "[$model] FAIL(finalize)"
  else
    echo "[$model] FAIL(missing shards — see $LOG/${model}_sh*.log)"
  fi
}

run2d openai_clip 8 &
run2d biomedclip  8 &
run2d medsiglip   8 &
wait

echo "######## aggregating $(date) ########"
"$CRIMSON" inspect_pheno/aggregate.py > "$LOG/aggregate.log" 2>&1 \
  && echo "[aggregate] OK — inspect_pheno/inspect_pheno_summary.md" \
  || echo "[aggregate] FAIL — see $LOG/aggregate.log"
echo "######## INSPECT-phenotype baselines DONE $(date) ########"

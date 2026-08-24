#!/bin/bash
# PARALLEL INSPECT-phenotype zero-shot — fills the single 96GB GH200 by running all
# baselines CONCURRENTLY (the 2D encode is single-threaded, so it is also sharded across
# processes). 72 cores available. Native original pipeline per model. f-VLM excluded.
#   nohup ./run_inspect_pheno_par.sh > _logs_inspect_pheno/run.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"
export CLEAR3D_REPO=${ACT_REPO:-/path/to/ACT}/analysis
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2   # keep per-proc CPU modest under heavy concurrency
LOG=_logs_inspect_pheno; mkdir -p "$LOG"
MERLIN=python
CTCLIP=python
ML=python
CRIMSON=python
DS=inspect_pheno

echo "######## PARALLEL INSPECT-phenotype START $(date) on $(hostname) ########"

# ---------- 3D natives: concurrent, DataLoader-parallel (CPU resample feeds GPU) ----------
( "$MERLIN" merlin/run_zeroshot.py  --datasets $DS --source native --batch_size 8 --num_workers 12 \
    > "$LOG/merlin.log"  2>&1 && echo "[merlin] OK"  || echo "[merlin] FAIL" ) &
( "$CTCLIP" m3dclip/run_zeroshot.py --datasets $DS --source native --batch_size 8 --num_workers 12 \
    > "$LOG/m3dclip.log" 2>&1 && echo "[m3dclip] OK" || echo "[m3dclip] FAIL" ) &
sleep 5
( "$CTCLIP" ctclip/run_zeroshot.py  --datasets $DS --source native --batch_size 2 --num_workers 8 \
    > "$LOG/ctclip.log"  2>&1 && echo "[ctclip] OK"  || echo "[ctclip] FAIL" ) &

# ---------- 2D sharded: N processes encode disjoint blocks in parallel, then finalize ----------
run2d () {  # model  nshards
  local model="$1" N="$2" i rc=0
  for i in $(seq 0 $((N-1))); do
    "$ML" run_zeroshot_2d.py --model "$model" --datasets $DS --nshards "$N" --shard "$i" \
        > "$LOG/${model}_sh${i}.log" 2>&1 &
    sleep 1
  done
  wait
  "$ML" run_zeroshot_2d.py --model "$model" --datasets $DS --nshards "$N" --finalize \
      > "$LOG/${model}_final.log" 2>&1 && echo "[$model] OK" || echo "[$model] FAIL"
}
sleep 8
run2d openai_clip 4 &
run2d biomedclip  4 &
sleep 4
run2d medsiglip   6 &

wait
echo "######## aggregating $(date) ########"
"$CRIMSON" inspect_pheno/aggregate.py > "$LOG/aggregate.log" 2>&1 \
  && echo "[aggregate] OK — inspect_pheno/inspect_pheno_summary.md" \
  || echo "[aggregate] FAIL — see $LOG/aggregate.log"
echo "######## INSPECT-phenotype baselines DONE $(date) ########"

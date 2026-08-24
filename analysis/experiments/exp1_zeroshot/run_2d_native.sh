#!/bin/bash
# Re-run the 2D baselines (OpenAI-CLIP / BiomedCLIP / MedSigLIP) on exp1 zero-shot
# under the OFFICIAL native pipeline:
#   native NIfTI scans (native.py) -> Google 3-window RGB (MedGemma high-dim-CT:
#   R wide[-1024,1024] / G soft-tissue[-135,215] / B narrow[0,80]) -> uniform
#   ~85-slice sampling -> each model's own 2D processor -> avg-pool. mode=native.
#
# SHARDED one process per (model, dataset) = 15 procs, to parallelize the
# CPU-bound per-slice preprocessing + NIfTI I/O and keep the GH200 fed. Per-proc
# CPU threads capped so 15 procs don't thrash the cores. Same datasets across
# models share the OS page cache (read in load_split order).
#
# Detached:  nohup ./run_2d_native.sh > _logs_2d_native/run_all.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"
ROOT=${ACT_REPO:-/path/to/ACT}/analysis
for _envf in "$ROOT/../.env" "$ROOT/.env"; do [[ -f "$_envf" ]] && { set -a; . "$_envf"; set +a; break; }; done  # HF_TOKEN for gated MedSigLIP
PY=python
export CT2D_WINDOWS="${CT2D_WINDOWS:-3win}"
export CT2D_NSLICES="${CT2D_NSLICES:-85}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"  # cap per-proc CPU threads (15 procs)
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
DS="ctrate_test radchest rsna2023_test pmbb_chest_nc pmbb_abd_ce"
LOG=_logs_2d_native; mkdir -p "$LOG"
echo "==== 2D native rerun START @ $(date) | windows=$CT2D_WINDOWS slices=$CT2D_NSLICES omp=$OMP_NUM_THREADS ===="

for m in openai_clip biomedclip medsiglip; do
  for ds in $DS; do
    ( echo "[$m $ds] start @ $(date)"
      "$PY" run_zeroshot_2d.py --model "$m" --source native --datasets "$ds"
      echo "[$m $ds] DONE @ $(date)" ) > "$LOG/${m}_${ds}.log" 2>&1 &
  done
done

wait
echo "==== 2D native rerun FULLY DONE @ $(date) ===="
echo "re-aggregate:  cd $ROOT && python experiments/exp1_zeroshot/aggregate_results.py"

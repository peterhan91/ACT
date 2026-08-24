#!/bin/bash
# End-to-end INSPECT-phenotype zero-shot for all baselines on the single GH200.
# Serialized (one model at a time) + resilient (one failure never blocks the rest).
# Native original-pipeline per model; 2,612 test vols × 221 phecodes (matches ours).
# f-VLM intentionally EXCLUDED: its mask-conditioned per-organ design only defines
# zero-shot for organ-tied chest findings (hardcoded 18-label organ map), not 221
# arbitrary EHR phecodes. See STATUS.md.
#   nohup ./run_inspect_pheno_all.sh > _logs_inspect_pheno/run.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")"
export CLEAR3D_REPO=${ACT_REPO:-/path/to/ACT}/analysis
LOG=_logs_inspect_pheno; mkdir -p "$LOG"
MERLIN=python
CTCLIP=python
ML=python
CRIMSON=python
DS="inspect_pheno"

run () {  # name  logfile  cmd...
  local name="$1"; shift
  local lg="$LOG/$1.log"; shift
  echo "==== [$name] START $(date +%F\ %H:%M:%S) ===="
  if "$@" > "$lg" 2>&1; then
    echo "[$name] OK $(date +%F\ %H:%M:%S)"
  else
    echo "[$name] FAIL (exit $?) — see $lg $(date +%F\ %H:%M:%S)"
  fi
}

echo "######## INSPECT-phenotype baselines START $(date) on $(hostname) ########"

# 2D (env ml311): light -> medsiglip heaviest last
run openai_clip  openai_clip  "$ML" run_zeroshot_2d.py --model openai_clip --datasets $DS
run biomedclip   biomedclip   "$ML" run_zeroshot_2d.py --model biomedclip  --datasets $DS

# 3D native (env merlin / ctclip): m3d light, merlin medium, ctclip heavy grid
run m3dclip      m3dclip      "$CTCLIP" m3dclip/run_zeroshot.py --datasets $DS --source native
run merlin       merlin       "$MERLIN" merlin/run_zeroshot.py  --datasets $DS --source native
run ctclip       ctclip       "$CTCLIP" ctclip/run_zeroshot.py  --datasets $DS --source native

# medsiglip last (448px tower over every slice ~ slowest)
run medsiglip    medsiglip    "$ML" run_zeroshot_2d.py --model medsiglip --datasets $DS

echo "######## aggregating $(date) ########"
"$CRIMSON" inspect_pheno/aggregate.py > "$LOG/aggregate.log" 2>&1 \
  && echo "[aggregate] OK — see inspect_pheno/inspect_pheno_summary.md" \
  || echo "[aggregate] FAIL — see $LOG/aggregate.log"

echo "######## INSPECT-phenotype baselines DONE $(date) ########"

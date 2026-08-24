#!/bin/bash
# Build all 4 baseline aarch64 envs SEQUENTIALLY and IDEMPOTENTLY.
# Designed to run detached via nohup and to be safe to re-run:
#   * an env already finished (marker _envlogs/<m>.done) is skipped;
#   * a partial env prefix from an interrupted run is removed and rebuilt clean;
#   * one failure does not block the others.
# Per-env logs: _envlogs/<m>.log   Rolling status: _envlogs/STATUS.txt
cd "$(dirname "$0")"
mkdir -p _envlogs
STATUS=_envlogs/STATUS.txt
echo "==== build_all start @ $(date) (pid $$) ====" >> "$STATUS"

for m in merlin fvlm ctclip; do
  if [ -f "_envlogs/$m.done" ]; then
    echo "$m SKIP (already done) @ $(date)" >> "$STATUS"
    continue
  fi
  PREFIX="/path/to/conda/envs/$m"
  if [ -d "$PREFIX" ]; then
    echo "$m: removing partial prefix @ $(date)" >> "$STATUS"
    rm -rf "$PREFIX"
  fi
  echo "==== building $m @ $(date) ====" >> "$STATUS"
  if bash "$m/setup_env.sh" > "_envlogs/$m.log" 2>&1; then
    touch "_envlogs/$m.done"
    echo "$m OK @ $(date)" >> "$STATUS"
  else
    echo "$m FAILED @ $(date) (tail: $(tail -n1 _envlogs/$m.log 2>/dev/null))" >> "$STATUS"
  fi
done
echo "==== ALL DONE @ $(date) ====" >> "$STATUS"

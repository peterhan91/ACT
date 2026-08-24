#!/bin/bash
# CT-RATE-18 LR sweep for v1 / linear_<LLM>.
# Usage:  bash run_v1_llm_lr_sweep_ctrate18.sh <LLM>          where LLM ∈ {sfr,harrier,openai}
# Writes per-LR test results to outputs/v1/cbm/linear_<LLM>__lr<LR>__test_results.json
# and the picked LR (by best val AUC) to outputs/v1/cbm/linear_<LLM>__bestlr.txt
set -e

LLM="${1:?missing LLM (sfr|harrier|openai)}"

export CLIP3D_RUN_TAG=v1
export CLIP3D_BEST=clip_3d_ctrate_merlin_v1
export PY=python

cd "${ACT_REPO:-/path/to/ACT}/analysis"

LOG_DIR=logs/v1
mkdir -p "$LOG_DIR"

LRS=(1.0 1e-1 3e-2 1e-2 5e-3 3e-3 1e-3 5e-4 2e-4 1e-4 5e-5)

echo "[ctrate18-lr-sweep/${LLM}] start  $(date)"
for LR in "${LRS[@]}"; do
  echo "=== [ctrate18-lr-sweep/${LLM}] lr=$LR  $(date) ==="
  $PY exp_linear_ct.py \
      --llms "$LLM" \
      --lr "$LR" \
      --out_suffix "__lr${LR}" \
      2>&1 | tee "$LOG_DIR/ctrate18_lr_sweep_${LLM}_${LR}.log"
done

echo "[ctrate18-lr-sweep/${LLM}] building summary..."
LLM="$LLM" $PY - <<'PYEOF'
import json, glob, os, re
LLM = os.environ["LLM"]
rows = []
for f in sorted(glob.glob(f'outputs/v1/cbm/linear_{LLM}__lr*__test_results.json')):
    base = os.path.basename(f)
    m = re.match(rf'linear_{LLM}__lr(.+)__test_results\.json', base)
    if not m: continue
    lr = m.group(1)
    with open(f) as fh: d = json.load(fh)
    rows.append((lr, d.get('test_mean_auc'), d.get('best_val_auc'),
                 d.get('n_test'), d.get('epochs_trained')))
rows.sort(key=lambda r: float(r[0]))
out_csv = f'outputs/v1/cbm/linear_{LLM}__lr_sweep_summary.csv'
with open(out_csv, 'w') as out:
    out.write('lr,test_mean_auc,best_val_auc,n_test,epochs\n')
    for r in rows: out.write(','.join(str(x) for x in r) + '\n')
print(open(out_csv).read())
# pick by best val AUC, tie-break by test AUC (matches existing openai sweep behavior)
best = max(rows, key=lambda r: (r[2], r[1]))
print(f'[best] lr={best[0]}  val_auc={best[2]:.4f}  test_auc={best[1]:.4f}')
with open(f'outputs/v1/cbm/linear_{LLM}__bestlr.txt','w') as f: f.write(best[0])
PYEOF

echo "[ctrate18-lr-sweep/${LLM}] done  $(date)"

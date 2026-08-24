#!/bin/bash
# INSPECT PE-3 LR sweep for v1 / linear_<LLM>.
# Usage:  bash run_v1_llm_lr_sweep_pe.sh <LLM>          where LLM ∈ {sfr,harrier,openai}
set -e

LLM="${1:?missing LLM (sfr|harrier|openai)}"

export CLIP3D_RUN_TAG=v1
export CLIP3D_BEST=clip_3d_ctrate_merlin_v1
export PY=python

cd "${ACT_REPO:-/path/to/ACT}/analysis"

LOG_DIR=logs/v1
mkdir -p "$LOG_DIR"

LRS=(1.0 1e-1 3e-2 1e-2 5e-3 3e-3 1e-3 5e-4 2e-4 1e-4 5e-5)

echo "[pe-lr-sweep/${LLM}] start  $(date)"
for LR in "${LRS[@]}"; do
  echo "=== [pe-lr-sweep/${LLM}] lr=$LR  $(date) ==="
  $PY exp_phenotype_ct.py \
      --labelsets pe \
      --methods "linear_${LLM}" \
      --lr "$LR" \
      --out_suffix "__lr${LR}" \
      2>&1 | tee "$LOG_DIR/pe_lr_sweep_${LLM}_${LR}.log"
done

echo "[pe-lr-sweep/${LLM}] building summary..."
LLM="$LLM" $PY - <<'PYEOF'
import json, glob, os, re
LLM = os.environ["LLM"]
rows = []
for f in sorted(glob.glob(f'outputs/v1/external/pe_pheno__test__linear_{LLM}__lr*__results.json')):
    base = os.path.basename(f)
    m = re.match(rf'pe_pheno__test__linear_{LLM}__lr(.+)__results\.json', base)
    if not m: continue
    lr = m.group(1)
    with open(f) as fh: d = json.load(fh)
    rows.append((lr, d.get('mean_auc'), d.get('n_volumes'), d.get('n_labels')))
rows.sort(key=lambda r: float(r[0]))
out_csv = f'outputs/v1/external/pe_pheno__linear_{LLM}__lr_sweep_summary.csv'
with open(out_csv, 'w') as out:
    out.write('lr,test_mean_auc,n_volumes,n_labels\n')
    for r in rows: out.write(','.join(str(x) for x in r) + '\n')
print(open(out_csv).read())
best = max(rows, key=lambda r: r[1])
print(f'[best] lr={best[0]}  test_auc={best[1]:.4f}')
with open(f'outputs/v1/external/pe_pheno__linear_{LLM}__bestlr.txt','w') as f: f.write(best[0])
PYEOF

echo "[pe-lr-sweep/${LLM}] done  $(date)"

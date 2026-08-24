#!/bin/bash
# INSPECT-221 phenotype LR sweep for v1 / linear_<LLM>.
# Usage:  bash run_v1_llm_lr_sweep_pheno.sh <LLM>          where LLM ∈ {sfr,harrier,openai}
# Picks best LR by validation mean AUC (matches the openai phenotype sweep convention).
set -e

LLM="${1:?missing LLM (sfr|harrier|openai)}"

export CLIP3D_RUN_TAG=v1
export CLIP3D_BEST=clip_3d_ctrate_merlin_v1
export PY=python

cd ${ACT_REPO:-/path/to/ACT}/analysis

LOG_DIR=logs/v1
mkdir -p "$LOG_DIR"

LRS=(1.0 1e-1 3e-2 1e-2 5e-3 3e-3 1e-3 5e-4 2e-4 1e-4 5e-5)

echo "[pheno-lr-sweep/${LLM}] start  $(date)"
for LR in "${LRS[@]}"; do
  echo "=== [pheno-lr-sweep/${LLM}] lr=$LR  $(date) ==="
  $PY exp_phenotype_ct.py \
      --labelsets phenotype \
      --methods "linear_${LLM}" \
      --lr "$LR" \
      --out_suffix "__lr${LR}" \
      2>&1 | tee "$LOG_DIR/pheno_lr_sweep_${LLM}_${LR}.log"
done

echo "[pheno-lr-sweep/${LLM}] building summary..."
LLM="$LLM" $PY - <<'PYEOF'
import json, glob, os, re
LLM = os.environ["LLM"]
# collect (lr → split → mean_auc)
by_lr = {}
for f in sorted(glob.glob(f'outputs/v1/external/phenotype__*__linear_{LLM}__lr*__results.json')):
    base = os.path.basename(f)
    m = re.match(rf'phenotype__(train|valid|test)__linear_{LLM}__lr(.+)__results\.json', base)
    if not m: continue
    split, lr = m.group(1), m.group(2)
    with open(f) as fh: d = json.load(fh)
    by_lr.setdefault(lr, {})[split] = (d.get('mean_auc'), d.get('n_volumes'), d.get('n_labels'))

rows = []
for lr in sorted(by_lr.keys(), key=lambda x: float(x)):
    for split in ('train', 'valid', 'test'):
        if split in by_lr[lr]:
            auc, n, nl = by_lr[lr][split]
            rows.append((lr, split, auc, n, nl))
out_csv = f'outputs/v1/external/phenotype__linear_{LLM}__lr_sweep_summary.csv'
with open(out_csv, 'w') as out:
    out.write('lr,split,mean_auc,n_volumes,n_labels\n')
    for r in rows: out.write(','.join(str(x) for x in r) + '\n')
print(open(out_csv).read())

# pick best by validation AUC
val_rows = [(lr, by_lr[lr]['valid'][0]) for lr in by_lr if 'valid' in by_lr[lr]]
best_lr, best_val = max(val_rows, key=lambda r: r[1])
test_auc = by_lr[best_lr]['test'][0] if 'test' in by_lr[best_lr] else float('nan')
print(f'[best] lr={best_lr}  val_auc={best_val:.4f}  test_auc={test_auc:.4f}')
with open(f'outputs/v1/external/phenotype__linear_{LLM}__bestlr.txt','w') as f: f.write(best_lr)
PYEOF

echo "[pheno-lr-sweep/${LLM}] done  $(date)"

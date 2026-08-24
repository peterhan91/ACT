#!/bin/bash
# =============================================================================
# exp4 — GENERALIZED phenotype linear-probe trainer for any LLM concept-embedding
# backend: LR sweep -> best LR (by valid AUC) -> 20 seeds -> aggregate -> 20-seed
# concept audit. Trains off the cached llm_repr.inspect_{train,valid,test}.<llm>.npy.
#
# Usage:  nohup bash experiments/exp4_confounder_audit/run_pheno_probe.sh f2llm \
#              > experiments/exp4_confounder_audit/_logs/f2llm/run.log 2>&1 & disown
# =============================================================================
set -uo pipefail
cd "${ACT_REPO:-/path/to/ACT}/analysis"
LLM="${1:?usage: run_pheno_probe.sh <LLM>  (e.g. gteqwen2, f2llm)}"
export CLIP3D_RUN_TAG=v1
export CLIP3D_BEST=clip_3d_ctrate_merlin_v1
export SKIP_INLINE_AUDIT=1
PY=python
LOGD="experiments/exp4_confounder_audit/_logs/${LLM}"
mkdir -p "$LOGD"
LRS=(1.0 1e-1 3e-2 1e-2 5e-3 3e-3 1e-3 5e-4 2e-4 1e-4 5e-5)

echo "################ [1/4] LR sweep | phenotype linear_${LLM} | $(date) ################"
for LR in "${LRS[@]}"; do
  echo "=== sweep lr=$LR  $(date) ==="
  $PY exp_phenotype_ct.py --labelsets phenotype --methods "linear_${LLM}" \
      --lr "$LR" --out_suffix "__lr${LR}" 2>&1 | tee "$LOGD/sweep_${LR}.log"
done

echo "################ [2/4] pick best LR by validation AUC | $(date) ################"
LLM="$LLM" $PY - <<'PYEOF'
import json, glob, os, re
LLM = os.environ["LLM"]
by_lr = {}
for f in sorted(glob.glob(f'outputs/v1/external/phenotype__*__linear_{LLM}__lr*__results.json')):
    m = re.match(rf'phenotype__(train|valid|test)__linear_{LLM}__lr(.+)__results\.json', os.path.basename(f))
    if not m:
        continue
    by_lr.setdefault(m.group(2), {})[m.group(1)] = json.load(open(f)).get('mean_auc')
out = f'outputs/v1/external/phenotype__linear_{LLM}__lr_sweep_summary.csv'
with open(out, 'w') as fh:
    fh.write('lr,split,mean_auc\n')
    for lr in sorted(by_lr, key=float):
        for sp in ('train', 'valid', 'test'):
            if sp in by_lr[lr]:
                fh.write(f'{lr},{sp},{by_lr[lr][sp]}\n')
print(open(out).read())
val = [(lr, by_lr[lr]['valid']) for lr in by_lr if 'valid' in by_lr[lr]]
best, bv = max(val, key=lambda r: r[1])
print(f'[best] lr={best}  val={bv:.4f}  test={by_lr[best].get("test", float("nan"))}')
open(f'outputs/v1/external/phenotype__linear_{LLM}__bestlr.txt', 'w').write(best)
PYEOF
BEST=$(cat "outputs/v1/external/phenotype__linear_${LLM}__bestlr.txt")
[[ -z "$BEST" ]] && { echo "ERROR: no best LR selected" >&2; exit 1; }
echo "  >>> best LR = $BEST"

echo "################ [3/4] 20 seeds @ lr=$BEST | $(date) ################"
for S in $(seq 1 20); do
  echo "=== seed=$S lr=$BEST  $(date) ==="
  $PY exp_phenotype_ct.py --labelsets phenotype --methods "linear_${LLM}" \
      --lr "$BEST" --seed "$S" --out_suffix "__seed${S}__lr${BEST}" \
      2>&1 | tee "$LOGD/seed${S}.log"
done

echo "################ [4/4] aggregate + 20-seed concept audit | $(date) ################"
$PY aggregate_20seeds.py --llm "$LLM" --pheno_bestlr "$BEST" 2>&1 | tee "$LOGD/aggregate.log"
# NOTE: the legacy concept_audit_20seeds.py below (analysis/concept_audit_20seeds.py in
# this release; invoked relative to the analysis/ working directory set above) produces
# cosine-similarity rankings that are used ONLY as subset-selection input. The
# authoritative Eq. 2 alignment-score exports are export_f2llm_adam_top25.py and
# build_f2llm_natural_clinical_audit.py in this directory.
$PY concept_audit_20seeds.py --task phenotype --llm "$LLM" --bestlr "$BEST" 2>&1 | tee "$LOGD/audit.log"
echo "################ DONE ($LLM) | $(date) ################"

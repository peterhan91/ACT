#!/bin/bash
# Task-1 (INSPECT phenotype-221, canon v1 + F2LLM concept-LLM embeddings) re-run
# with the LBFGS solver, plus the 20-seed concept-DISTRIBUTION as a TRAIN BOOTSTRAP.
#
# Mirror of run_v1_phenotype_lbfgs.sh (which did this for OpenAI). Under Adam, f2llm
# was the best embedding model (0.7087 vs openai 0.7060); openai+LBFGS then jumped to
# 0.7200. This run checks whether the LBFGS gain also lifts f2llm past openai.
#
# Why LBFGS: a linear sigmoid/BCE multi-label head has no cross-label sharing, so it
# is N independent per-label logistic regressions either way; LBFGS solves each to the
# exact deterministic L2 optimum. Why bootstrap: the fixed INSPECT train/val split makes
# a convex LBFGS fit deterministic, so the 20 "seeds" resample the TRAIN patients to
# yield a real concept distribution (data/sampling uncertainty, not optimizer jitter).
#
# CPU/many-core bound by design (sklearn LBFGS + joblib n_jobs=-1, single-thread BLAS) —
# saturates the GH200 Grace cores. The f2llm inspect-split llm_repr is already cached,
# so no GPU projection is needed. Waits for any other heavy probe job first.

PY=python
cd ${ACT_REPO:-/path/to/ACT}/analysis || exit 1
export CLIP3D_RUN_TAG=v1
export CLIP3D_BEST=clip_3d_ctrate_merlin_v1
# single-thread BLAS so the per-label joblib fits scale linearly across cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SKIP_INLINE_AUDIT=1   # canonical concept importance comes from the 20-seed audit

LLM=f2llm
LOG=logs/v1/phenotype_lbfgs_${LLM}
mkdir -p "$LOG"
EXT=outputs/v1/external
AUD=outputs/v1/audit
LRTAG=lbfgs
SEEDS=$(seq 1 20)

echo "=== [phenotype LBFGS ${LLM}] START $(date) ==="

# 0. Wait for any other heavy probe / purification job to finish (avoid 72-core contention).
echo "  [0] waiting for any running probe/purification job..."
while pgrep -f "run_purification_eval_f2llm.sh|run_v1_phenotype_lbfgs_f2llm.sh" >/dev/null 2>&1; do sleep 30; done
echo "      clear to proceed  $(date)"

# 0b. Preserve the Adam-based canonical f2llm aggregates that aggregate/audit will overwrite.
#     (per-seed Adam coefs are __lr3e-2__ and do NOT collide with __lrlbfgs__.)
BK="$AUD/_adam_backup"; mkdir -p "$BK"
for f in "$AUD/phenotype__linear_${LLM}__concept_importance.csv" \
         "$AUD/phenotype__linear_${LLM}__concept_importance.json" \
         "$AUD/phenotype__linear_${LLM}__concept_importance__20seeds_meta.json" \
         "$EXT/phenotype__linear_${LLM}__20seeds_summary.csv" \
         "$EXT/phenotype__linear_${LLM}__20seeds_overall.json"; do
    [ -f "$f" ] && cp -n "$f" "$BK/$(basename "$f").adam"
done
echo "  [0b] backed up Adam f2llm aggregates -> $BK/*.adam"

# 1. Canonical point estimate: full-train LBFGS fit (headline performance).
echo "  [1] full-train LBFGS probe (point estimate)  $(date)"
$PY exp_phenotype_ct.py --labelsets phenotype --methods linear_${LLM} \
    --solver lbfgs --lbfgs_C 1.0 --lbfgs_max_iter 2000 \
    --min_test_positives 50 --out_suffix __${LRTAG} \
    > "$LOG/00_point_estimate.log" 2>&1
echo "      point-estimate exit=$?  $(date)"
if [ ! -f "$EXT/phenotype__linear_${LLM}__${LRTAG}__coefs.pt" ]; then
    echo "  ABORT: point estimate produced no coefs; see $LOG/00_point_estimate.log" >&2
    exit 1
fi

# 2. 20-seed concept distribution = LBFGS on bootstrapped TRAIN sets.
for S in $SEEDS; do
    echo "  [2] bootstrap seed $S  $(date)"
    $PY exp_phenotype_ct.py --labelsets phenotype --methods linear_${LLM} \
        --solver lbfgs --lbfgs_C 1.0 --lbfgs_max_iter 2000 --bootstrap --seed "$S" \
        --min_test_positives 50 --out_suffix __seed${S}__lr${LRTAG} \
        > "$LOG/seed${S}.log" 2>&1
done
echo "      20 bootstrap seeds done  $(date)"

# 3. Performance distribution across seeds (mean ± std).
echo "  [3] aggregate_20seeds  $(date)"
$PY aggregate_20seeds.py --llm ${LLM} --pheno_bestlr ${LRTAG} \
    > "$LOG/agg.log" 2>&1
echo "      aggregate exit=$?"

# 4. Concept distribution (canonical concept_importance + __20seeds_meta.json).
# NOTE: concept_audit_20seeds.py is the LEGACY normalized-cosine implementation.
# Its output here is subset-selection input only, never an Eq. 2 alignment score.
echo "  [4] concept_audit_20seeds (concept distribution)  $(date)"
$PY concept_audit_20seeds.py --task phenotype --llm ${LLM} --bestlr ${LRTAG} \
    --center_concepts --top_k_per_label 100 --top_k_stability 10 \
    > "$LOG/audit.log" 2>&1
echo "      audit exit=$?"

echo "=== [phenotype LBFGS ${LLM}] DONE $(date) ==="

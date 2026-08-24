#!/bin/bash
# Actionable concept-level disease phenotyping — purification eval, F2LLM embeddings.
# Mirror of run_purification_eval.sh (openai) for the f2llm concept-LLM space.
# Task 2 (purified probe) + Task 3 (concept-importance audit + with/without comparison)
# on the SAME radiologist-curated 86-label trusted concept spaces — the refinement
# rule is regex-over-concept-text, so it is embedding-agnostic and reused as-is.
# Only the f2llm concept embeddings + the f2llm full-bank ranking differ.
#
# INSPECT CTPA + EHR phenotypes, model = canon v1 + F2LLM concept-LLM embeddings.
# Faithful to the LBFGS purification setup (deterministic). CPU/many-core bound
# (sklearn LBFGS + multi-threaded BLAS for the f2llm projection; the audit uses
# joblib n_jobs=-1). --device cuda is passed for any GPU-capable ops.

PY=python
cd ${ACT_REPO:-/path/to/ACT}/analysis || exit 1
export CLIP3D_RUN_TAG=v1
LLM=f2llm
LOG=logs/v1/purification_${LLM}
mkdir -p "$LOG"
OUT=outputs/v1/external/trusted_concept_probe_${LLM}_lbfgs
RANK=outputs/v1/audit/phenotype__linear_${LLM}__concept_importance.json   # f2llm ranking for original_topk

echo "=== [purification eval ${LLM}] START $(date) ==="

# 0. Wait for the f2llm 20-seed LBFGS job to finish: frees the 72 Grace cores AND
#    produces the LBFGS f2llm concept ranking ($RANK) that original_topk needs.
echo "  [0] waiting for f2llm 20-seed job (run_v1_phenotype_lbfgs_f2llm.sh)..."
while pgrep -f "run_v1_phenotype_lbfgs_f2llm.sh" >/dev/null 2>&1; do sleep 30; done
echo "      f2llm 20-seed job done; proceeding  $(date)"
echo "      original_topk ranking source: $RANK"
ls -l "$RANK" 2>/dev/null || { echo "  ABORT: $RANK missing (f2llm concept audit step did not finish)"; exit 1; }

# 0b. Preserve any prior f2llm purification result for provenance.
if [ -d "$OUT" ] && [ ! -d "${OUT}.prev" ]; then
    echo "  [0b] backing up old result: $OUT -> ${OUT}.prev"
    mv "$OUT" "${OUT}.prev"
fi
mkdir -p "$OUT"

# 1. Purified probe (Task 2) + matched with/without arms (Task 3 comparison):
#      original_topk = WITHOUT purification (f2llm's raw top-100 concepts, from $RANK)
#      trusted       = WITH purification (radiologist-defensible cleaned subspace)
#      direct        = direct-tier only (strictest)
echo "  [1/3] trusted_concept_probe (original_topk + trusted + direct)  $(date)"
$PY trusted_concept_probe.py \
    --methods original_topk trusted direct \
    --feature_spaces ${LLM} \
    --f2llm_concept_bank concept_bank.f2llm_emb.npz \
    --audit_json "$RANK" \
    --solver lbfgs --lbfgs_C 1e6 --lbfgs_max_iter 5000 \
    --max_concepts 0 --min_test_positives 50 \
    --device cuda --out_dir "$OUT" \
    > "$LOG/01_probe.log" 2>&1
echo "      probe exit=$?  $(date)"

# 2. Concept-importance audit (Task 3a): signed per-concept importance for the
#    purified f2llm probes (verifies trustworthy concept-driven prediction). Non-fatal.
echo "  [2/3] audit (concept importance, --llm ${LLM})  $(date)"
$PY _audit_trusted_openai_probes.py --llm ${LLM} > "$LOG/02_audit.log" 2>&1
echo "      audit exit=$?  $(date)"

# 3. With/without comparison table (Task 3b).
echo "  [3/3] make_trusted_vs_baseline (--llm ${LLM})  $(date)"
$PY make_trusted_vs_baseline.py --llm ${LLM} > "$LOG/03_compare.log" 2>&1
echo "      compare exit=$?  $(date)"

echo "=== [purification eval ${LLM}] DONE $(date) ==="

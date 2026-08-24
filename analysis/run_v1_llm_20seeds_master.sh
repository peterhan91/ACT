#!/bin/bash
# Master pipeline for v1 / linear_<LLM> 20-seed evaluation:
#   1) CT-RATE-18 LR sweep → outputs/v1/cbm/linear_<LLM>__bestlr.txt
#   2) INSPECT PE-3 LR sweep → outputs/v1/external/pe_pheno__linear_<LLM>__bestlr.txt
#   3) INSPECT-221 phenotype LR sweep → outputs/v1/external/phenotype__linear_<LLM>__bestlr.txt
#   4) 20-seed runs across all 3 labelsets at each labelset's best LR
#      (concept-importance audits are produced per seed by exp_phenotype_ct.py)
#   5) aggregate → mean ± std summary CSVs
#
# Usage:  bash run_v1_llm_20seeds_master.sh <LLM>          where LLM ∈ {sfr,harrier,openai}
set -e

LLM="${1:?missing LLM (sfr|harrier|openai)}"

cd ${ACT_REPO:-/path/to/ACT}/analysis
mkdir -p logs/v1

echo "[master/${LLM}] $(date)  starting v1/linear_${LLM} 20-seed pipeline"

echo "[master/${LLM}] step 1/5 — CT-RATE-18 LR sweep"
bash run_v1_llm_lr_sweep_ctrate18.sh "$LLM"

echo "[master/${LLM}] step 2/5 — PE-3 LR sweep"
bash run_v1_llm_lr_sweep_pe.sh "$LLM"

echo "[master/${LLM}] step 3/5 — phenotype-221 LR sweep"
bash run_v1_llm_lr_sweep_pheno.sh "$LLM"

echo "[master/${LLM}] step 4/5 — 20 seeds × 3 labelsets at best LRs (with per-seed audit)"
bash run_v1_llm_20seeds.sh "$LLM"

echo "[master/${LLM}] $(date)  done."

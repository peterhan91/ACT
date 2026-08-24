#!/bin/bash
set -e
export PY=python
export SCRIPT=${ACT_REPO:-/path/to/ACT}/analysis/get_concepts_ct.py
export URL=http://localhost:8000/v1
export MODEL="Qwen/Qwen3.5-35B-A3B"
echo "=========================================="
echo "[$(date)] CT-RATE Findings_EN start (22,773 unique)"
echo "=========================================="
$PY $SCRIPT \
  --reports_csv ${ACT_REPO:-/path/to/ACT}/model/data/ct_rate/train_reports.csv \
  --text_col Findings_EN \
  --vllm_url $URL --model "$MODEL" \
  --out ${ACT_REPO:-/path/to/ACT}/analysis/ctrate_concepts.jsonl \
  --concurrency 128
echo "=========================================="
echo "[$(date)] MERLIN Findings_EN start (~25,477 unique, full radiologist findings)"
echo "=========================================="
$PY $SCRIPT \
  --reports_csv ${ACT_REPO:-/path/to/ACT}/model/data/merlin/original_reports.csv \
  --text_col Findings_EN \
  --vllm_url $URL --model "$MODEL" \
  --out ${ACT_REPO:-/path/to/ACT}/analysis/merlin_concepts.jsonl \
  --concurrency 128
echo "[$(date)] DONE."

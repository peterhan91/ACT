#!/bin/bash
# Submit all six preprocessing jobs (3 datasets x 2 stages) with proper
# afterok dependencies so each stage-2 (NIfTI -> h5) only runs after its
# stage-1 (source -> NIfTI + split CSVs) finishes successfully.
#
# Datasets run in parallel (no cross-dataset dependency); the three stage-1
# jobs are submitted simultaneously, and each stage-2 waits on its own
# stage-1 only.
#
# Usage:
#   bash preprocess_new/run_scripts/submit_all.sh
# Optional: pass dataset names to submit a subset, e.g.:
#   bash preprocess_new/run_scripts/submit_all.sh multimodalpe rsna2023

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/path/to/logs"
mkdir -p "$LOG_DIR"

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(multimodalpe rsna2023 coca)
fi

# stage-1 filename suffix differs per dataset (npy vs dicom source)
declare -A STAGE1_NAME=(
    [multimodalpe]="multimodalpe_npy2nifti.sh"
    [rsna2023]="rsna2023_dicom2nifti.sh"
    [coca]="coca_dicom2nifti.sh"
)
declare -A STAGE2_NAME=(
    [multimodalpe]="multimodalpe_preprocess_all.sh"
    [rsna2023]="rsna2023_preprocess_all.sh"
    [coca]="coca_preprocess_all.sh"
)

declare -A JOB1
declare -A JOB2

echo "=== submitting stage-1 jobs ==="
for ds in "${DATASETS[@]}"; do
    s1="$SCRIPT_DIR/${STAGE1_NAME[$ds]}"
    if [ ! -f "$s1" ]; then
        echo "ERROR: stage-1 script not found: $s1" >&2
        exit 1
    fi
    jid=$(sbatch --parsable "$s1")
    JOB1[$ds]=$jid
    echo "  $ds  stage-1: $s1  ->  jobid $jid"
done

echo
echo "=== submitting stage-2 jobs (afterok:<stage-1>) ==="
for ds in "${DATASETS[@]}"; do
    s2="$SCRIPT_DIR/${STAGE2_NAME[$ds]}"
    if [ ! -f "$s2" ]; then
        echo "ERROR: stage-2 script not found: $s2" >&2
        exit 1
    fi
    dep="afterok:${JOB1[$ds]}"
    jid=$(sbatch --parsable --dependency="$dep" "$s2")
    JOB2[$ds]=$jid
    echo "  $ds  stage-2: $s2  ->  jobid $jid  (waits on ${JOB1[$ds]})"
done

echo
echo "=== summary ==="
printf "%-14s %-12s %-12s\n" "dataset" "stage-1" "stage-2"
for ds in "${DATASETS[@]}"; do
    printf "%-14s %-12s %-12s\n" "$ds" "${JOB1[$ds]}" "${JOB2[$ds]}"
done
echo
echo "Watch progress with:    squeue -u \$USER"
echo "Tail a stage-1 log:     tail -f $LOG_DIR/<jobname>_<jobid>.out"
echo "Cancel everything:      scancel ${JOB1[*]} ${JOB2[*]}"

#!/bin/bash
# End-to-end full-res fVLM reproduction on CT-RATE (matches authors' default TotalSegmentator):
#   1) parallel full-res masks (W sharded workers, roi_subset=4 organs) -> masks_fullres/
#   2) fvlm eval on those masks -> results_fullres/
#   3) compare vs the fast-mask baseline (results/ , mean 0.628)
# Parallel across cases on the single GH200 (each TS proc is independent). Resumable.
set -uo pipefail
cd "$(dirname "$0")"
TSPY=python
FVLMPY=python
CRIMSON=python
W=${W:-10}
DS=${DS:-ctrate_test}
mkdir -p logs_fullres

echo "===== [fullres] masks: $W parallel workers on $DS | $(date) ====="
pids=()
for r in $(seq 0 $((W-1))); do
  # MUST be a SHORT path: nnUNet's SyncManager binds an AF_UNIX socket here (~108 char limit).
  wtmp="/dev/shm/fvlm_r${r}"; rm -rf "$wtmp"; mkdir -p "$wtmp"
  TMPDIR="$wtmp" OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
  "$TSPY" gen_masks_fullres.py --dataset "$DS" --rank "$r" --world "$W" \
     > "logs_fullres/mask_${DS}_r${r}.log" 2>&1 &
  pids+=($!)
  sleep 2   # small stagger so GPU init doesn't spike simultaneously (models already cached)
done
echo "mask worker pids: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "===== [fullres] masks DONE (fail=$fail) | $(date) ====="
echo "fullres masks count: $(ls masks_fullres/ 2>/dev/null | wc -l)"

echo "===== [fullres] eval | $(date) ====="
FVLM_MASKS_SUFFIX=_fullres FVLM_RESULTS_DIR="$(pwd)/results_fullres" \
  "$FVLMPY" run_zeroshot.py --datasets "$DS"

echo "===== [fullres] compare fast vs full-res | $(date) ====="
"$CRIMSON" - <<'PY'
import json
from pathlib import Path
def load(p):
    f=Path(p)
    return json.load(open(f)) if f.exists() else None
fast=load("results/ctrate_test__native__results.json")
full=load("results_fullres/ctrate_test__native__results.json")
lung=['Emphysema','Atelectasis','Lung nodule','Lung opacity','Pulmonary fibrotic sequela',
      'Pleural effusion','Mosaic attenuation pattern','Peribronchial thickening','Consolidation',
      'Bronchiectasis','Interlobular septal thickening']
import statistics as st
def summ(d,tag):
    if not d: print(f"{tag}: MISSING"); return
    pl=d['per_label_auc']; nl=[k for k in pl if k not in lung]
    print(f"{tag}: mean={d['mean_auc']:.4f}  n={d['n_volumes']}  "
          f"lung={st.mean(pl[c] for c in lung if c in pl):.4f}  "
          f"nonlung={st.mean(pl[c] for c in nl):.4f}")
summ(fast,"fast  ")
summ(full,"fullres")
if fast and full:
    print(f"\nDelta mean = {full['mean_auc']-fast['mean_auc']:+.4f}")
    print("per-label (fast -> fullres):")
    for c in fast['per_label_auc']:
        a=fast['per_label_auc'][c]; b=full['per_label_auc'].get(c)
        if b is not None:
            mark=" <lung>" if c in lung else ""
            print(f"  {c:34s} {a:.4f} -> {b:.4f}  ({b-a:+.4f}){mark}")
PY
echo "===== [fullres] ALL DONE | $(date) ====="

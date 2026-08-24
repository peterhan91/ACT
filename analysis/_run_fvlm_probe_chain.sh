#!/bin/bash
# f-VLM CT-RATE in-domain probe: extract per-organ features (GPU, sharded) -> merge ->
# build 3 pooled feature variants -> lr-sweep -> 20-seed, joining the SAME fairA/perseed
# pipeline the other baselines use. masks_ctrate_train is 47121/47146 (99.95%; 25
# volumes permanently fail inside TotalSegmentator itself with a JSONDecodeError, even
# with threading pinned to 1 - looks like bad source data, not a resource issue).
# extract_feats.py already zero-fills rows whose mask is missing (present=0), so this
# does not block anything - those 25 rows just carry no f-VLM signal.
#
# Shard count from a live smoke test (2026-07-02): 8 concurrent extract_feats.py shards
# measured ~2.7-3.7GB GPU mem each (worst case ~3.7GB) and ~2 CPU cores each on an
# otherwise-idle 98GB/72-core node. 24 shards x ~3.8GB ~= 91GB (leaves ~7GB headroom,
# full use of the GPU without cutting it close) and 24 x 2 = 48 CPU cores (GPU, not
# CPU, is the binding constraint for this stage).
#
# Detached: nohup ./_run_fvlm_probe_chain.sh > logs/v1/fvlm_probe_chain/_driver.log 2>&1 & disown
set -u
export CLIP3D_RUN_TAG=v1
ROOT=${ACT_REPO:-/path/to/ACT}/analysis
FV=python
PY=python
cd "$ROOT/experiments/exp1_zeroshot/fvlm"
LOG=$ROOT/logs/v1/fvlm_probe_chain; mkdir -p "$LOG"
OUT=$ROOT/outputs/v1/probe_compare/fairA
PSOUT=$OUT/perseed; mkdir -p "$PSOUT"
LRS="1e-1 3e-2 1e-2 5e-3"
NSEEDS=20
NSH=24
TAGS="fvlm_concat1024 fvlm_meanpool256 fvlm_vit768"
ts(){ date '+%F %T'; }

echo "[$(ts)] === f-VLM CT-RATE probe chain START ==="

have_masks_test=$(find masks -iname "*.nii.gz" 2>/dev/null | wc -l)
have_masks_train=$(find masks_ctrate_train -iname "*.nii.gz" 2>/dev/null | wc -l)
echo "[$(ts)] masks: ctrate_test=$have_masks_test/2489  ctrate_train=$have_masks_train/47146"
[ "$have_masks_test" -ge 2000 ] || { echo "[$(ts)] FATAL: ctrate_test masks incomplete, abort"; exit 1; }
[ "$have_masks_train" -ge 40000 ] || { echo "[$(ts)] FATAL: ctrate_train masks incomplete, abort"; exit 1; }

extract(){  # $1=dataset $2=nshards
  local DS=$1 N=$2
  if [ -f "feats_probe/${DS}__organ_proj256.npy" ]; then
    echo "[$(ts)] extract_feats $DS: merged output already present, skipping (resume)"
    return
  fi
  echo "[$(ts)] extract_feats $DS ($N shards, full GPU width)"
  for s in $(seq 0 $((N - 1))); do
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      "$FV" extract_feats.py --dataset "$DS" --nshards "$N" --shard "$s" \
      > "$LOG/extract_${DS}_shard${s}.log" 2>&1 &
  done
  wait
  local bad=0
  for s in $(seq 0 $((N - 1))); do
    grep -q "^\[extract\] $DS block" "$LOG/extract_${DS}_shard${s}.log" || {
      echo "[$(ts)] FATAL: shard $s/$N for $DS did not complete cleanly"
      tail -n 20 "$LOG/extract_${DS}_shard${s}.log"; bad=1; }
  done
  [ "$bad" -eq 0 ] || exit 1
  echo "[$(ts)] extract_feats $DS: all $N shards confirmed complete; merging"
  "$FV" extract_feats.py --dataset "$DS" --merge > "$LOG/merge_${DS}.log" 2>&1
  grep -q "^\[merge\]" "$LOG/merge_${DS}.log" || {
    echo "[$(ts)] FATAL: merge $DS failed"; cat "$LOG/merge_${DS}.log"; exit 1; }
  echo "[$(ts)] merge $DS done: $(tail -n1 "$LOG/merge_${DS}.log")"
}

extract ctrate_test "$NSH"
extract ctrate_train "$NSH"

echo "[$(ts)] build_probe_feats (concat1024 / meanpool256 / vit768)"
"$FV" build_probe_feats.py --dataset ctrate_test  > "$LOG/build_ctrate_test.log" 2>&1
"$FV" build_probe_feats.py --dataset ctrate_train > "$LOG/build_ctrate_train.log" 2>&1
cat "$LOG/build_ctrate_test.log" "$LOG/build_ctrate_train.log"
for TAG in $TAGS; do
  [ -f "feats_probe/ctrate_train__${TAG}.npy" ] && [ -f "feats_probe/ctrate_test__${TAG}.npy" ] || {
    echo "[$(ts)] FATAL: build_probe_feats did not produce $TAG"; exit 1; }
done

# ---- Stage 1: lr sweep x 3 pooled feature variants, in-domain ctrate_train->ctrate_test ----
# Full-width: 3 variants x 4 LRs = 12 jobs all at once, NTH=6 -> 72 cores (=nproc).
for TAG in $TAGS; do
  TR="feats_probe/ctrate_train__${TAG}.npy"
  TE="feats_probe/ctrate_test__${TAG}.npy"
  for LR in $LRS; do
    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 \
      "$PY" "$ROOT/probe_features.py" --mode indomain \
        --train_dataset ctrate_train --feat_train "$TR" \
        --dataset ctrate_test --feat "$TE" \
        --model_tag "$TAG" --lr "$LR" --device cpu \
        --out "$OUT/ctrate_test__${TAG}__lr${LR}.json" \
        > "$LOG/sweep_${TAG}_lr${LR}.log" 2>&1 &
  done
done
wait
echo "[$(ts)] lr sweep done (3 variants x 4 LRs = 12 jobs)"

fails=0
for f in "$LOG"/sweep_fvlm_*.log; do
  grep -q "mean AUROC" "$f" || { echo "[$(ts)] SWEEP JOB FAILED: $f"; tail -n 15 "$f"; fails=$((fails+1)); }
done
[ "$fails" -eq 0 ] || echo "[$(ts)] WARNING: $fails/12 sweep jobs failed"

"$PY" - "$OUT" "$TAGS" <<'PYEOF'
import json, glob, sys
out, tags = sys.argv[1], sys.argv[2].split()
for TAG in tags:
    fs = sorted(glob.glob(f"{out}/ctrate_test__{TAG}__lr*.json"))
    cand = [json.load(open(f)) for f in fs]
    if not cand:
        print(f"[{TAG}] NO sweep results"); continue
    best = max(cand, key=lambda d: d.get("val_auc") or -1)
    json.dump(best, open(f"{out}/ctrate_test__{TAG}.json", "w"), indent=2)
    sweepstr = " ".join(f"{c['lr']}:{c['mean_auc']:.4f}" for c in sorted(cand, key=lambda d: -(d.get('val_auc') or -1)))
    print(f"[{TAG}] best lr={best['lr']} val={best['val_auc']:.4f} test={best['mean_auc']:.4f} | {sweepstr}")
PYEOF
echo "[$(ts)] best-lr selection done"

# ---- Stage 2: 20-seed perseed at each variant's best lr ----
# Only 3 cells total (one per pooling variant); give each all the threads instead of
# throttling job count: 3 x NTH=24 = 72 cores.
for TAG in $TAGS; do
  best="$OUT/ctrate_test__${TAG}.json"
  [ -f "$best" ] || { echo "[$(ts)] SKIP perseed $TAG (no best-lr json)"; continue; }
  LR=$("$PY" -c "import json;print(json.load(open('$best'))['lr'])")
  echo "[$(ts)] launch perseed $TAG lr=$LR"
  OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 \
    "$PY" "$ROOT/perseed_probe.py" --model_tag "$TAG" --lr "$LR" --n_seeds $NSEEDS \
      --train_dataset ctrate_train --feat_train "feats_probe/ctrate_train__${TAG}.npy" \
      --dataset ctrate_test --feat "feats_probe/ctrate_test__${TAG}.npy" \
      --out "$PSOUT/ctrate__${TAG}" > "$LOG/perseed_${TAG}.log" 2>&1 &
done
wait
echo "[$(ts)] === ALL f-VLM perseed runs finished ==="

fails=0
for f in "$LOG"/perseed_fvlm_*.log; do
  grep -q "seeds  test AUROC" "$f" || { echo "[$(ts)] PERSEED JOB FAILED: $f"; tail -n 15 "$f"; fails=$((fails+1)); }
done
[ "$fails" -eq 0 ] || echo "[$(ts)] WARNING: $fails/3 perseed jobs failed"

# ---- regenerate the unified _perseed_summary.csv ----
"$PY" - "$PSOUT" <<'PYEOF'
import json, glob, sys, csv, os
out=sys.argv[1]; rows=[]
for f in sorted(glob.glob(f"{out}/*__*.json")):
    d=json.load(open(f)); short=os.path.basename(f).split("__")[0]
    t=d["test_mean_auc"]
    rows.append((short, d["model"], d["in_dim"], d["lr"], d["n_seeds"],
                 t["mean"], t["std"], d["n"], d["n_labels"]))
order={"ctrate":0,"rsna":1,"inspect":2}
rows.sort(key=lambda r:(order.get(r[0],9), -r[5]))
with open(f"{out}/_perseed_summary.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["dataset","model","in_dim","lr","n_seeds",
                                  "test_mean_auc_mean","test_mean_auc_std","n_test","n_labels"])
    for r in rows: w.writerow(r)
print(f"\n{'dataset':8} {'model':16} {'lr':>6} {'seeds':>5} {'TEST_AUROC(mean+-std)':>22}")
for sh,m,d,lr,ns,mu,sd,n,L in rows:
    print(f"{sh:8} {m:16} {str(lr):>6} {ns:5d}   {mu:.4f} +- {sd:.4f}")
print(f"\n-> {out}/_perseed_summary.csv")
PYEOF
echo "[$(ts)] === f-VLM CT-RATE probe chain COMPLETE ==="

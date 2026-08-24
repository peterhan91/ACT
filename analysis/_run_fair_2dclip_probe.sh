#!/bin/bash
# Add the 3 2D-CLIP baselines (openai_clip, biomedclip, medsiglip) to the fairA
# in-domain probe (CT-RATE/RSNA/INSPECT-221), now that all 3 have complete native
# ctrate_train/rsna2023_train/inspect_pheno_train imgfeat caches (biomedclip's
# ctrate_train finished last, 2026-07-02 01:13). Follows the fairA
# lr-sweep -> best-by-val -> 20-seed protocol used for every added model, but no encode stage is needed here (features already cached) and all 3
# models run together since none need GPU. Pure CPU throughout.
#
# Model tags keep the _clip/native names (openai_clip/biomedclip/medsiglip) - do NOT
# confuse with fairA's existing "openai" tag, which is the UNRELATED ours-pipeline
# concept->LLM projection using OpenAI text embeddings for the concept bank.
#
# Detached: nohup ./_run_fair_2dclip_probe.sh > logs/v1/fair_2dclip_probe/_driver.log 2>&1 & disown
set -u
export CLIP3D_RUN_TAG=v1
ROOT=${ACT_REPO:-/path/to/ACT}/analysis
PY=python
cd "$ROOT"
CACHE2D=experiments/_cache2d
OUT=outputs/v1/probe_compare/fairA
PSOUT=$OUT/perseed
LOG=logs/v1/fair_2dclip_probe
mkdir -p "$OUT" "$PSOUT" "$LOG"
LRS="1e-1 3e-2 1e-2 5e-3"
NSEEDS=20
MODELS="openai_clip biomedclip medsiglip"
ts(){ date '+%F %T'; }

fpath(){  # $1=model $2=dataset -> feature npy path
  case "$2" in
    inspect_pheno|inspect_pheno_train|inspect_pheno_valid)
      echo "$CACHE2D/$1__$2__imgfeat.npy" ;;
    *)
      echo "$CACHE2D/$1__$2__native_imgfeat.npy" ;;
  esac
}

echo "[$(ts)] === 2D-CLIP baselines fairA probe START ($MODELS) ==="

missing=0
for M in $MODELS; do
  for D in ctrate_train ctrate_test rsna2023_train rsna2023_test inspect_pheno_train inspect_pheno_valid inspect_pheno; do
    f=$(fpath "$M" "$D")
    [ -f "$f" ] || { echo "[$(ts)] MISSING $M/$D -> $f"; missing=1; }
  done
done
[ "$missing" -eq 0 ] || { echo "[$(ts)] FATAL: missing feature files, abort"; exit 1; }
echo "[$(ts)] all feature files verified present"

# ---- Stage 1: lr sweep, in-domain (val carved from train for ctrate/rsna; explicit val for inspect) ----
# Full-width: 3 models x 3 datasets x 4 LRs = 36 jobs, all launched at once, NTH=2 each
# -> 72 cores (=nproc), no throttle needed since the job count itself is the cap.
NTH=2
sweep(){  # $1=tag $2=train_ds $3=test_ds $4=valid_ds(optional)
  local TAG=$1 TRD=$2 TED=$3 VAD=${4:-}
  for M in $MODELS; do
    local TR TE VAARG=""
    TR=$(fpath "$M" "$TRD"); TE=$(fpath "$M" "$TED")
    if [ -n "$VAD" ]; then VAARG="--valid_dataset $VAD --feat_valid $(fpath "$M" "$VAD")"; fi
    for LR in $LRS; do
      OMP_NUM_THREADS=$NTH MKL_NUM_THREADS=$NTH OPENBLAS_NUM_THREADS=$NTH \
        $PY probe_features.py --mode indomain \
          --train_dataset "$TRD" --feat_train "$TR" $VAARG \
          --dataset "$TED" --feat "$TE" \
          --model_tag "$M" --lr "$LR" --device cpu \
          --out "$OUT/${TAG}__${M}__lr${LR}.json" \
          > "$LOG/sweep_${TAG}_${M}_lr${LR}.log" 2>&1 &
    done
  done
}
sweep inspect_pheno inspect_pheno_train inspect_pheno inspect_pheno_valid
sweep rsna2023 rsna2023_train rsna2023_test
sweep ctrate_test ctrate_train ctrate_test
wait
echo "[$(ts)] lr sweep done (3 models x 3 datasets x 4 LRs = 36 jobs)"

fails=0
for f in "$LOG"/sweep_*.log; do
  grep -q "mean AUROC" "$f" || { echo "[$(ts)] SWEEP JOB FAILED: $f"; tail -n 15 "$f"; fails=$((fails+1)); }
done
[ "$fails" -eq 0 ] || echo "[$(ts)] WARNING: $fails/36 sweep jobs failed (continuing with whatever succeeded)"

# ---- pick best lr per (tag, model) by val AUROC ----
$PY - "$OUT" "$MODELS" <<'PYEOF'
import json, glob, sys
out, models = sys.argv[1], sys.argv[2].split()
for tag in ["ctrate_test", "rsna2023", "inspect_pheno"]:
    for M in models:
        fs = sorted(glob.glob(f"{out}/{tag}__{M}__lr*.json"))
        cand = [json.load(open(f)) for f in fs]
        if not cand:
            print(f"[{tag}/{M}] NO sweep results"); continue
        best = max(cand, key=lambda d: d.get("val_auc") or -1)
        json.dump(best, open(f"{out}/{tag}__{M}.json", "w"), indent=2)
        sweepstr = " ".join(f"{c['lr']}:{c['mean_auc']:.4f}" for c in sorted(cand, key=lambda d: -(d.get('val_auc') or -1)))
        print(f"[{tag}/{M}] best lr={best['lr']} val={best['val_auc']:.4f} test={best['mean_auc']:.4f} | {sweepstr}")
PYEOF
echo "[$(ts)] best-lr selection done"

# ---- Stage 2: 20-seed perseed at each model's best lr ----
# Full-width: only 9 cells total (3 models x 3 datasets), each running 20 seeds
# SEQUENTIALLY in-process (perseed_probe.py has no internal seed-parallelism) -> give
# each cell more threads instead of throttling job count: 9 x NTH=8 = 72 cores.
NTH=8
launch_perseed(){  # $1=short $2=sweep_tag $3=model $4=train_ds $5=test_ds $6=valid_ds(optional)
  local SH=$1 TAG=$2 M=$3 TRD=$4 TED=$5 VAD=${6:-}
  local best="$OUT/${TAG}__${M}.json"
  [ -f "$best" ] || { echo "[$(ts)] SKIP perseed $SH/$M (no best-lr json)"; return; }
  local LR VAARG=""
  LR=$($PY -c "import json;print(json.load(open('$best'))['lr'])")
  local TR TE; TR=$(fpath "$M" "$TRD"); TE=$(fpath "$M" "$TED")
  if [ -n "$VAD" ]; then VAARG="--valid_dataset $VAD --feat_valid $(fpath "$M" "$VAD")"; fi
  echo "[$(ts)] launch perseed $SH/$M lr=$LR"
  OMP_NUM_THREADS=$NTH MKL_NUM_THREADS=$NTH OPENBLAS_NUM_THREADS=$NTH \
    $PY perseed_probe.py --model_tag "$M" --lr "$LR" --n_seeds $NSEEDS \
      --train_dataset "$TRD" --feat_train "$TR" $VAARG \
      --dataset "$TED" --feat "$TE" \
      --out "$PSOUT/${SH}__${M}" > "$LOG/perseed_${SH}_${M}.log" 2>&1 &
}
for M in $MODELS; do launch_perseed inspect inspect_pheno "$M" inspect_pheno_train inspect_pheno inspect_pheno_valid; done
for M in $MODELS; do launch_perseed rsna    rsna2023      "$M" rsna2023_train    rsna2023_test; done
for M in $MODELS; do launch_perseed ctrate  ctrate_test   "$M" ctrate_train      ctrate_test; done
wait
echo "[$(ts)] === ALL 2D-CLIP perseed runs finished ==="

fails=0
for f in "$LOG"/perseed_*.log; do
  grep -q "seeds  test AUROC" "$f" || { echo "[$(ts)] PERSEED JOB FAILED: $f"; tail -n 15 "$f"; fails=$((fails+1)); }
done
[ "$fails" -eq 0 ] || echo "[$(ts)] WARNING: $fails/9 perseed jobs failed"

# ---- regenerate the unified _perseed_summary.csv (now incl these 3 + whatever was already there) ----
$PY - "$PSOUT" <<'PYEOF'
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
echo "[$(ts)] === 2D-CLIP baselines fairA probe chain ALL DONE ==="

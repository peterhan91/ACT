#!/bin/bash
# 20-random-seed rerun of the Option-A fair in-domain probe, saving head weights +
# test predictions for every seed (perseed_probe.py). CT-RATE / INSPECT / RSNA only,
# no k-fold. Each (dataset, model) trained at its already-selected best LR (by val
# AUROC, from outputs/v1/probe_compare/fairA/_fairA{,_external}_summary.csv).
#
#   ours model : ours_raw3dclip (raw 3D-CLIP backbone) + f2llm/openai (concept->LLM)
#   3D baseline: merlin, ctclip
#   2D baseline: openai_clip/biomedclip/medsiglip -- NOT here yet (in-domain feats
#                still caching; need their own LR sweep first). Same driver later.
# Cells whose features are missing (e.g. CT-RATE ctclip, still encoding) auto-skip.
set -u
export CLIP3D_RUN_TAG=v1
ROOT=${ACT_REPO:-/path/to/ACT}/analysis
PY=python
cd "$ROOT"
OUT=outputs/v1/probe_compare/fairA/perseed; mkdir -p "$OUT"
LOG="$OUT/logs"; mkdir -p "$LOG"
CACHE=outputs/v1/cache
MR=experiments/exp1_zeroshot/merlin/results
CR=experiments/exp1_zeroshot/ctclip/results
NSEEDS=20
MAXJOBS=6; NTH=4

fpath(){  # $1=model $2=dataset_name -> echo feature path
  case "$1" in
    merlin)         echo "$MR/$2__native__imgfeat.npy";;
    ctclip)         echo "$CR/$2__native__imgfeat.npy";;
    ours_raw3dclip) echo "$CACHE/img_feats.$2.npy";;
    f2llm)          echo "$CACHE/llm_repr.$2.f2llm.npy";;
    openai)         echo "$CACHE/llm_repr.$2.openai.npy";;
  esac
}

# best LR per (short, model) -- the published best-by-val choices.
bestlr(){ case "$1/$2" in
  inspect/ours_raw3dclip) echo 0.03;; inspect/merlin) echo 0.03;; inspect/ctclip) echo 0.005;;
  inspect/f2llm) echo 0.03;; inspect/openai) echo 0.1;;
  ctrate/ours_raw3dclip) echo 0.005;; ctrate/merlin) echo 0.03;; ctrate/ctclip) echo 0.005;;
  ctrate/f2llm) echo 0.03;; ctrate/openai) echo 0.1;;
  rsna/ours_raw3dclip) echo 0.1;; rsna/merlin) echo 0.1;; rsna/ctclip) echo 0.03;;
  rsna/f2llm) echo 0.1;; rsna/openai) echo 0.1;;
esac; }

launch(){  # $1=short $2=model $3=train_ds $4=test_ds $5=valid_ds(optional "")
  local SH=$1 M=$2 TRD=$3 TED=$4 VAD=$5
  local TR TE VA="" VAARG="" LR
  TR=$(fpath "$M" "$TRD"); TE=$(fpath "$M" "$TED"); LR=$(bestlr "$SH" "$M")
  if [ ! -f "$TR" ] || [ ! -f "$TE" ]; then echo "SKIP $SH/$M (missing feats: $TR | $TE)"; return; fi
  if [ -n "$VAD" ]; then VA=$(fpath "$M" "$VAD"); VAARG="--valid_dataset $VAD --feat_valid $VA"; fi
  while [ "$(jobs -rp | wc -l)" -ge $MAXJOBS ]; do sleep 2; done
  echo "[$(date +%T)] launch $SH/$M lr=$LR (train=$TRD test=$TED ${VAD:+valid=$VAD})"
  OMP_NUM_THREADS=$NTH MKL_NUM_THREADS=$NTH OPENBLAS_NUM_THREADS=$NTH \
    $PY perseed_probe.py --model_tag "$M" --lr "$LR" --n_seeds $NSEEDS \
      --train_dataset "$TRD" --feat_train "$TR" $VAARG \
      --dataset "$TED" --feat "$TE" \
      --out "$OUT/${SH}__${M}" > "$LOG/${SH}__${M}.log" 2>&1 &
}

echo "=== perseed in-domain (n_seeds=$NSEEDS, $MAXJOBS jobs x $NTH cores) START $(date) ==="
# INSPECT (explicit train/valid/test)
for M in ours_raw3dclip merlin ctclip f2llm openai; do
  launch inspect "$M" inspect_pheno_train inspect_pheno inspect_pheno_valid
done
# RSNA-trauma (carve val from train)
for M in ours_raw3dclip merlin ctclip f2llm openai; do
  launch rsna "$M" rsna2023_train rsna2023_test ""
done
# CT-RATE (carve val from train; ctclip auto-skips until its train feats finish encoding)
for M in ours_raw3dclip merlin ctclip f2llm openai; do
  launch ctrate "$M" ctrate_train ctrate_test ""
done
wait
echo "=== ALL perseed cells done $(date) ==="

# ---- aggregate the 20-seed test AUROC mean±std per (dataset, model) ----
$PY - "$OUT" <<'PYEOF'
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
print(f"\n{'dataset':8} {'model':16} {'lr':>6} {'seeds':>5} {'TEST_AUROC(mean±std)':>22}")
for sh,m,d,lr,ns,mu,sd,n,L in rows:
    print(f"{sh:8} {m:16} {str(lr):>6} {ns:5d}   {mu:.4f} ± {sd:.4f}")
print(f"\n-> {out}/_perseed_summary.csv")
PYEOF
echo "DONE."

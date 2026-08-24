#!/usr/bin/env python3
"""20-random-seed rerun of the Option-A fair in-domain linear probe, SAVING every
trained head's weights + every test-prediction set (the data the published fairA
JSONs threw away -- they kept only the single-seed mean AUROC).

For one (dataset, model) at its chosen LR, this trains the IDENTICAL Adam head
(probe_features.train_head: same epochs/patience/wd/bs/val_frac/early-stop-on-val)
exactly as `probe_features.py --mode indomain --seed s` would, once per seed
s in 1..n_seeds. Each rerun varies torch/np init + minibatch order (and, when no
explicit valid set is given, the val carve) -- i.e. optimisation stochasticity ->
the 20-dot spread + per-case predictions for bootstrap CIs / weight attribution.

Only CT-RATE / INSPECT / RSNA in-domain (no k-fold). Each model probed on its OWN
embedding; feature paths supplied by the caller (see _run_perseed_indomain.sh).

Outputs (next to --out):
  <out>.json   summary: per-seed val/test AUROC arrays + mean/std, metadata
  <out>.npz    full archive (savez_compressed):
      seeds          (S,)            the seed used for each rerun
      W              (S, k, d) f32   head weight per seed
      b              (S, k)    f32   head bias per seed
      probs          (S, n, k) f32   TEST predictions (sigmoid) per seed
      per_label_auc  (S, k)    f32   TEST per-label AUROC per seed
      val_auc        (S,)            best val AUROC per seed
      test_mean_auc  (S,)            macro TEST AUROC per seed  <- the 20 dots
      y              (n, k)    i8     TEST ground truth
      h5_idx         (n,)      i64    per-case id (aligns to the cached features)
      labels         (k,)      str
      model dataset train_dataset feat lr  (0-d scalars)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import probe_features as P  # reuse the EXACT head + training loop
C = P.C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="test set, e.g. ctrate_test / inspect_pheno / rsna2023_test")
    ap.add_argument("--feat", required=True, help="npy feats aligned to --dataset")
    ap.add_argument("--train_dataset", required=True)
    ap.add_argument("--feat_train", required=True)
    ap.add_argument("--valid_dataset", default=None, help="explicit val set (INSPECT); else carve from train per seed")
    ap.add_argument("--feat_valid", default=None)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--lr", type=float, required=True, help="the best LR already selected by val sweep")
    ap.add_argument("--out", required=True, help="output stem; writes <out>.json + <out>.npz")
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1, help="first seed; uses seed0 .. seed0+n_seeds-1")
    # head hyper-params: defaults identical to probe_features.py
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--weight_decay", type=float, default=1e-8)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # ---- load eval (test) + train, exactly like probe_features.indomain ----
    _, h5_idx_eval, y, labels, _ = C.load_dataset(args.dataset)
    y = y.astype(np.float32)
    X = np.load(args.feat).astype(np.float32)
    assert len(X) == len(y), f"feat {X.shape} vs y {y.shape} for {args.dataset}"

    _, _, ytr, _, _ = C.load_dataset(args.train_dataset)
    ytr = ytr.astype(np.float32)
    Xtr = np.load(args.feat_train).astype(np.float32)
    assert len(Xtr) == len(ytr), f"train feat {Xtr.shape} vs y {ytr.shape}"

    Xva = yva = None
    if args.valid_dataset and args.feat_valid:
        _, _, yva, _, _ = C.load_dataset(args.valid_dataset)
        yva = yva.astype(np.float32)
        Xva = np.load(args.feat_valid).astype(np.float32)
        assert len(Xva) == len(yva), f"valid feat {Xva.shape} vs y {yva.shape}"

    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))
    k, d, n = y.shape[1], X.shape[1], len(y)
    W = np.zeros((len(seeds), k, d), np.float32)
    b = np.zeros((len(seeds), k), np.float32)
    probs = np.zeros((len(seeds), n, k), np.float32)
    pla = np.zeros((len(seeds), k), np.float32)
    val_auc = np.zeros(len(seeds), np.float64)
    test_mean = np.zeros(len(seeds), np.float64)

    t0 = time.time()
    for i, s in enumerate(seeds):
        if Xva is not None:                                   # explicit val (INSPECT)
            m, bv = P.train_head(Xtr, ytr, Xva, yva, lr=args.lr, epochs=args.epochs,
                                 patience=args.patience, wd=args.weight_decay,
                                 bs=args.batch_size, seed=s, device=args.device)
        else:                                                 # carve val from train (CT-RATE/RSNA)
            tr, va = P.val_carve(np.arange(len(Xtr)), args.val_frac, s)
            m, bv = P.train_head(Xtr[tr], ytr[tr], Xtr[va], ytr[va], lr=args.lr, epochs=args.epochs,
                                 patience=args.patience, wd=args.weight_decay,
                                 bs=args.batch_size, seed=s, device=args.device)
        pr = P.predict(m, X, args.device)
        aucs = P.per_label_auc(pr, y)
        sd = m.state_dict()
        W[i] = sd["linear.weight"].cpu().numpy()
        b[i] = sd["linear.bias"].cpu().numpy()
        probs[i] = pr
        pla[i] = [aucs[j] if aucs[j] == aucs[j] else np.nan for j in range(k)]
        val_auc[i] = bv
        test_mean[i] = P.mean_auc(aucs)
        print(f"  [{args.model_tag}/{args.dataset}] seed {s:>2} "
              f"val={bv:.4f} test={test_mean[i]:.4f}", flush=True)

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_stem.with_suffix(".npz"),
        seeds=np.asarray(seeds, np.int64), W=W, b=b, probs=probs,
        per_label_auc=pla, val_auc=val_auc, test_mean_auc=test_mean,
        y=y.astype(np.int8), h5_idx=np.asarray(h5_idx_eval, np.int64),
        labels=np.asarray(list(labels)),
        model=np.asarray(args.model_tag), dataset=np.asarray(args.dataset),
        train_dataset=np.asarray(args.train_dataset), feat=np.asarray(args.feat),
        lr=np.asarray(args.lr, np.float64),
    )
    summary = {
        "model": args.model_tag, "dataset": args.dataset,
        "train_dataset": args.train_dataset, "feat": args.feat,
        "in_dim": int(d), "n": int(n), "n_labels": int(k), "labels": list(labels),
        "lr": args.lr, "n_seeds": len(seeds), "seeds": seeds,
        "test_mean_auc": {"mean": float(test_mean.mean()), "std": float(test_mean.std(ddof=1)),
                          "per_seed": [float(v) for v in test_mean]},
        "val_auc": {"mean": float(val_auc.mean()), "std": float(val_auc.std(ddof=1)),
                    "per_seed": [float(v) for v in val_auc]},
        "seconds": time.time() - t0,
    }
    json.dump(summary, open(out_stem.with_suffix(".json"), "w"), indent=2)
    print(f"[{args.model_tag}/{args.dataset}] {len(seeds)} seeds  "
          f"test AUROC {test_mean.mean():.4f}±{test_mean.std(ddof=1):.4f}  "
          f"({time.time()-t0:.0f}s) -> {out_stem.with_suffix('.npz')}")


if __name__ == "__main__":
    main()

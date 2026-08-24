#!/usr/bin/env python3
"""
Step 2b — Linear probe on the LLM-projected representation.

Mirrors CLEAR's `exp_linear.py`: train a logistic-regression head on the
4096-d (SFR) or 5376-d (Harrier) `llm_repr` features, hold out a 5% val split
from CT-RATE train, evaluate on CT-RATE test. Saves W/b and per-label AUCs.

The trained W vectors (in LLM space) are what `concept_importance.py` uses for
cosine-alignment audit — so this is the prerequisite for clinically-meaningful
top-concept lists.

Outputs (in outputs/cbm/):
    linear_<llm>__coefs.pt
    linear_<llm>__test_results.json
    linear_<llm>__test_probs.npy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from clear3d.data import load_split
from clear3d.features import encode_images, free_3d_clip
from clear3d.metrics import mean_auc, per_label_auc
from clear3d.paths import CBM_DIR, LABELS_18, cache_img_feats
from clear3d.projection import get_or_compute_llm_repr


class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def split_train_val(n: int, val_frac=0.05, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(int(n * val_frac), 1)
    return idx[n_val:], idx[:n_val]


@torch.no_grad()
def eval_loader(model, loader):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        logits = model(x.cuda())
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps, 0), np.concatenate(ys, 0)


def run_for_llm(llm: str, train_repr: np.ndarray, train_y: np.ndarray,
                test_repr: np.ndarray, test_y: np.ndarray,
                args) -> dict:
    print(f"\n=== Linear probe on llm_repr[{llm}]  (in_dim={train_repr.shape[1]}) ===")
    tr_idx, val_idx = split_train_val(len(train_repr), val_frac=0.05, seed=args.seed)

    tr_set = TensorDataset(torch.from_numpy(train_repr[tr_idx]).float(),
                           torch.from_numpy(train_y[tr_idx]).float())
    va_set = TensorDataset(torch.from_numpy(train_repr[val_idx]).float(),
                           torch.from_numpy(train_y[val_idx]).float())
    te_set = TensorDataset(torch.from_numpy(test_repr).float(),
                           torch.from_numpy(test_y).float())

    tr_loader = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=args.batch_size)
    te_loader = DataLoader(te_set, batch_size=args.batch_size)

    model = LinearHead(train_repr.shape[1], len(LABELS_18)).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    best_val = -np.inf
    best_state = None
    bad = 0
    t0 = time.time()
    history = []
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0; nb = 0
        for x, y in tr_loader:
            x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            loss = bce(model(x), y)
            loss.backward()
            optim.step()
            ep_loss += loss.item(); nb += 1
        train_loss = ep_loss / max(nb, 1)
        vp, vy = eval_loader(model, va_loader)
        v_auc = mean_auc(per_label_auc(vp, vy, LABELS_18))
        history.append((ep+1, train_loss, v_auc))
        if (ep + 1) % 10 == 0 or ep < 5:
            print(f"  ep{ep+1:3d}  loss={train_loss:.4f}  val_auc={v_auc:.4f}")
        if v_auc > best_val + 1e-5:
            best_val = v_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop ep{ep+1}  best_val={best_val:.4f}")
                break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    tp, ty = eval_loader(model, te_loader)
    auc_per = per_label_auc(tp, ty, LABELS_18)
    test_auc = mean_auc(auc_per)
    print(f"  trained {len(history)} epochs in {elapsed:.0f}s; test mean AUC = {test_auc:.4f}")

    suffix = getattr(args, "out_suffix", "") or ""
    out_coef = CBM_DIR / f"linear_{llm}{suffix}__coefs.pt"
    torch.save({
        "W": model.linear.weight.detach().cpu(),  # (n_labels, llm_dim)
        "b": model.linear.bias.detach().cpu(),
        "labels": LABELS_18,
        "llm": llm,
        "best_val_auc": best_val,
        "test_mean_auc": test_auc,
        "seed": int(getattr(args, "seed", -1)),
        "lr": float(args.lr),
    }, out_coef)
    np.save(CBM_DIR / f"linear_{llm}{suffix}__test_probs.npy", tp)
    with open(CBM_DIR / f"linear_{llm}{suffix}__test_results.json", "w") as f:
        json.dump({
            "method": f"linear_probe_{llm}",
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_repr)),
            "in_dim": int(train_repr.shape[1]),
            "epochs_trained": len(history),
            "best_val_auc": best_val,
            "test_mean_auc": test_auc,
            "test_per_label_auc": auc_per,
            "seconds": elapsed,
            "seed": int(getattr(args, "seed", -1)),
            "lr": float(args.lr),
        }, f, indent=2)
    return {"llm": llm, "test_mean_auc": test_auc,
            "test_per_label_auc": auc_per, "n": len(test_repr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llms", nargs="*", default=["sfr", "harrier", "openai"])
    ap.add_argument("--train_split", default="ctrate_train")
    ap.add_argument("--test_split", default="ctrate_test")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_suffix", default="",
                    help="appended to output filenames (e.g. __seed7__lr1e-1)")
    args = ap.parse_args()

    import random as _random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_df = load_split(args.train_split)
    test_df = load_split(args.test_split)
    train_y = train_df[LABELS_18].values.astype(np.float32)
    test_y = test_df[LABELS_18].values.astype(np.float32)

    # Make sure img_feats are cached, then project to LLM space
    encode_images(args.train_split, batch_size=4, num_workers=4)
    encode_images(args.test_split, batch_size=4, num_workers=4)
    free_3d_clip()

    rows = []
    for llm in args.llms:
        train_repr = get_or_compute_llm_repr(args.train_split, llm,
                                             np.load(cache_img_feats(args.train_split)))
        test_repr = get_or_compute_llm_repr(args.test_split, llm,
                                            np.load(cache_img_feats(args.test_split)))
        res = run_for_llm(llm, train_repr, train_y, test_repr, test_y, args)
        rows.append(res)

    suffix = args.out_suffix or ""
    summary = pd.DataFrame([
        {"llm": r["llm"], "test_mean_auc": r["test_mean_auc"], "n_test": r["n"],
         **r["test_per_label_auc"]} for r in rows])
    summary.to_csv(CBM_DIR / f"linear_summary{suffix}.csv", index=False)
    print("\n=== Linear-probe summary ===")
    print(summary[["llm", "test_mean_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()

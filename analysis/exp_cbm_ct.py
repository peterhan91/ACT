#!/usr/bin/env python3
"""
Step 2 — Concept Bottleneck Model.

Trains a single linear layer over the concept-similarity features (the
"bottleneck"). The forward is:

    concept_sim = img_feats @ clip_text_emb.T   # (B, M)
    logits      = concept_sim @ W + b           # (B, 18)
    probs       = sigmoid(logits)

We never materialize concept_sim for the full 47k train set (would be ~70 GB).
Instead we cache only the (47k, 768) image features and recompute concept_sim
per minibatch from `clip_text_emb` (kept on GPU once).

Outputs (in outputs/cbm/):
    coefs.pt              — {W, b, label_names, concepts} for downstream audit
    test_results.json
    test_probs.npy
    summary.csv           — train/val/test mean AUC + per-label AUC
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
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from clear3d.data import load_split
from clear3d.features import encode_images, free_3d_clip
from clear3d.metrics import mean_auc, per_label_auc
from clear3d.paths import CBM_DIR, LABELS_18
from clear3d.projection import load_concept_bank


def split_train_val(n: int, val_frac: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(int(n * val_frac), 1)
    return idx[n_val:], idx[:n_val]


class CBMLinear(nn.Module):
    def __init__(self, n_concepts: int, n_labels: int):
        super().__init__()
        self.linear = nn.Linear(n_concepts, n_labels, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, concept_sim: torch.Tensor) -> torch.Tensor:
        return self.linear(concept_sim)


def make_concept_sim(img_feats: torch.Tensor, clip_emb: torch.Tensor) -> torch.Tensor:
    """img_feats (B, 768) @ clip_emb (M, 768).T → (B, M). Both already L2-norm."""
    return img_feats @ clip_emb.T


@torch.no_grad()
def eval_loader(model: CBMLinear, loader: DataLoader, clip_emb: torch.Tensor,
                device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_y = [], []
    for img, y in loader:
        img = img.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            sims = make_concept_sim(img, clip_emb)
            logits = model(sims.float())
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.numpy())
    return np.concatenate(all_probs, 0), np.concatenate(all_y, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_split", default="ctrate_train")
    ap.add_argument("--test_split", default="ctrate_test")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-8)
    ap.add_argument("--encode_batch", type=int, default=4,
                    help="batch size when extracting 3D-CLIP image features")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------------------------------------------------------------- features
    print("\n=== 1. Encode train + test images (cached) ===")
    train_df = load_split(args.train_split)
    test_df = load_split(args.test_split)
    train_feats = encode_images(args.train_split,
                                batch_size=args.encode_batch,
                                num_workers=args.num_workers)
    test_feats = encode_images(args.test_split,
                               batch_size=args.encode_batch,
                               num_workers=args.num_workers)
    free_3d_clip()

    train_y = train_df[LABELS_18].values.astype(np.float32)
    test_y = test_df[LABELS_18].values.astype(np.float32)

    tr_idx, val_idx = split_train_val(len(train_df), val_frac=0.05, seed=args.seed)
    print(f"  train: {len(tr_idx)}   val: {len(val_idx)}   test: {len(test_df)}")

    # ----------------------------------------------------------------- bank
    print("\n=== 2. Load concept bank (clip_text only — CBM doesn't use LLM emb) ===")
    bank = load_concept_bank(llm="sfr", device="cpu")  # llm only used for shape; we keep clip_text
    clip_emb = bank["clip_text_emb"].to(device).to(torch.bfloat16)  # (M, 768)
    concepts = bank["concepts"]
    M = clip_emb.shape[0]
    print(f"  concepts on GPU: {M} × 768  bf16 ≈ {M*768*2/1e6:.0f} MB")
    del bank  # drop the (M, 4096) llm_emb we don't need

    # --------------------------------------------------------------- loaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_feats[tr_idx]).float(),
        torch.from_numpy(train_y[tr_idx]).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(train_feats[val_idx]).float(),
        torch.from_numpy(train_y[val_idx]).float(),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_feats).float(),
        torch.from_numpy(test_y).float(),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # ----------------------------------------------------------------- model
    model = CBMLinear(M, len(LABELS_18)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    print(f"\n=== 3. Train CBM linear ({M} concepts → {len(LABELS_18)} labels) ===")
    best_val = -np.inf
    best_state = None
    bad_epochs = 0
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        nb = 0
        for img, y in train_loader:
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                sims = make_concept_sim(img, clip_emb)
                logits = model(sims.float())
                loss = bce(logits, y)
            loss.backward()
            optim.step()
            ep_loss += loss.item()
            nb += 1
        train_loss = ep_loss / max(nb, 1)
        val_probs, val_y_arr = eval_loader(model, val_loader, clip_emb, device)
        val_auc_per = per_label_auc(val_probs, val_y_arr, LABELS_18)
        val_auc = mean_auc(val_auc_per)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_auc": val_auc})

        if epoch % 10 == 0 or epoch < 5:
            print(f"  ep{epoch+1:3d}  loss={train_loss:.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_val + 1e-5:
            best_val = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stop at epoch {epoch+1} (best_val={best_val:.4f})")
                break

    elapsed = time.time() - t0
    print(f"  trained in {elapsed:.0f}s ({len(history)} epochs)")

    model.load_state_dict(best_state)

    # -------------------------------------------------------------- evaluate
    print("\n=== 4. Evaluate test set ===")
    test_probs, test_y_arr = eval_loader(model, test_loader, clip_emb, device)
    test_auc_per = per_label_auc(test_probs, test_y_arr, LABELS_18)
    test_auc = mean_auc(test_auc_per)
    print(f"  test mean AUC = {test_auc:.4f}")

    train_probs, train_y_arr = eval_loader(model, DataLoader(train_dataset, batch_size=args.batch_size),
                                           clip_emb, device)
    train_auc = mean_auc(per_label_auc(train_probs, train_y_arr, LABELS_18))
    val_auc = mean_auc(per_label_auc(*eval_loader(model, val_loader, clip_emb, device), LABELS_18))

    # ----------------------------------------------------------------- save
    out_coef = CBM_DIR / "coefs.pt"
    torch.save({
        "W": model.linear.weight.detach().cpu(),  # (n_labels, n_concepts)
        "b": model.linear.bias.detach().cpu(),
        "labels": LABELS_18,
        "concepts": concepts,
        "best_val_auc": best_val,
        "test_mean_auc": test_auc,
    }, out_coef)
    print(f"  saved coefs → {out_coef}")

    np.save(CBM_DIR / "test_probs.npy", test_probs)
    with open(CBM_DIR / "test_results.json", "w") as f:
        json.dump({
            "train_split": args.train_split,
            "test_split": args.test_split,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_df)),
            "n_concepts": int(M),
            "n_labels": int(len(LABELS_18)),
            "labels": LABELS_18,
            "train_mean_auc": train_auc,
            "val_mean_auc": val_auc,
            "test_mean_auc": test_auc,
            "test_per_label_auc": test_auc_per,
            "epochs_trained": len(history),
            "seconds": elapsed,
            "args": vars(args),
        }, f, indent=2)

    summary_df = pd.DataFrame([
        {"dataset": args.test_split, "method": "cbm",
         "mean_auc": test_auc, **test_auc_per}
    ])
    summary_df.to_csv(CBM_DIR / "summary.csv", index=False)
    print(f"\n=== Done.  test mean AUC = {test_auc:.4f} ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
INSPECT phenotype-label evaluation.

Two label families are evaluated, both on the phenotype-manifest's
train / valid / test splits (NOT the older inspect_* h5 splits):

    A) phecodes filtered to >= 50 positives in the test split (fixed list,
       same set applied to train + valid).  Phecode strings (e.g. "Pleurisy;
       pleural effusion") are used as the label name for prompt embedding.

    B) the 3 native INSPECT PE tasks (pe_positive / pe_acute /
       pe_subsegmentalonly) joined onto the same manifest splits.

For each (labelset, split):
    - zeroshot: plain (3D-CLIP text tower), sfr, harrier
    - linear probe: train on phenotype-manifest TRAIN, eval on val + test
      (one head per LLM space). Trained head is also evaluated on the test
      set; AUCs reported per-split.

Image features and llm_repr are already cached per inspect_* h5 split. We
build a unified VolumeName → row index across all three caches and gather
the rows the manifest needs.

Outputs (in outputs/external/):
    phenotype__test__filtered_phecodes.csv    — the fixed phecode list
    phenotype__<split>__<labelset>__<method>__results.json + probs.npy
    phenotype__summary.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from clear3d.data import load_pe_split, load_split
from clear3d.features import encode_texts_clip, free_3d_clip, load_3d_clip
from clear3d.llm_prompts import encode_label_texts
from clear3d.metrics import mean_auc, per_label_auc, softmax_pos_neg
from clear3d.paths import (
    AUDIT_DIR, CACHE, EXTERNAL_DIR, PHENOTYPE_LABELS, PHENOTYPE_MANIFEST,
    PHENOTYPE_NAMES_CSV, cache_label_emb,
)
from clear3d.projection import get_or_compute_llm_repr, load_concept_bank


def _label_emb_path(llm: str, sig: str) -> Path:
    return cache_label_emb(llm).with_name(f"label_emb.{llm}.{sig}.npz")


def get_label_embeddings_for(labels: list[str], llm: str) -> dict:
    """Encode `<label>` and `no <label>` with the named LLM. Cached by hash."""
    import hashlib
    sig = hashlib.sha1("|".join(labels).encode("utf-8")).hexdigest()[:10]
    out = _label_emb_path(llm, sig)
    if out.exists():
        z = np.load(out, allow_pickle=True)
        return {"pos": z["pos"], "neg": z["neg"], "labels": list(z["labels"])}

    pos_texts = [lbl.lower() for lbl in labels]
    neg_texts = [f"no {lbl.lower()}" for lbl in labels]
    print(f"[label_emb:{llm}] encoding {2*len(labels)} prompts")
    all_emb = encode_label_texts(llm, pos_texts + neg_texts)
    pos = all_emb[: len(labels)]
    neg = all_emb[len(labels):]
    np.savez(out, pos=pos, neg=neg, labels=np.asarray(labels, dtype=object))
    return {"pos": pos, "neg": neg, "labels": labels}


# ---------------------------------------------------------------------------
# Build a unified VolumeName → (img_feats, llm_repr) lookup across the three
# inspect_* caches that already exist.
# ---------------------------------------------------------------------------
def build_inspect_lookup(extra_modes=()) -> dict:
    """Returns dict with keys: img (N×768), repr_sfr (N×4096), repr_harrier
    (N×5376), repr_openai (N×3072 if cached), repr_<m> for each extra mode,
    volume_to_row (VolumeName → int). Concatenates rows across
    inspect_train + inspect_valid + inspect_test. `extra_modes` (e.g.
    ("f2llm", "gteqwen2")) are loaded from their llm_repr cache when present,
    else projected on the fly via get_or_compute_llm_repr (needs the matching
    concept_bank.<m>_emb.npz)."""
    splits = ["inspect_train", "inspect_valid", "inspect_test"]
    img_blocks, sfr_blocks, har_blocks, oai_blocks, vol_lists = [], [], [], [], []
    extra_blocks = {m: [] for m in extra_modes}
    have_openai = all((CACHE / f"llm_repr.{s}.openai.npy").exists() for s in splits)
    for s in splits:
        idx = pd.read_csv(CACHE / f"volume_index.{s}.csv")
        img_s = np.load(CACHE / f"img_feats.{s}.npy")
        img_blocks.append(img_s)
        sfr_blocks.append(np.load(CACHE / f"llm_repr.{s}.sfr.npy"))
        har_blocks.append(np.load(CACHE / f"llm_repr.{s}.harrier.npy"))
        if have_openai:
            oai_blocks.append(np.load(CACHE / f"llm_repr.{s}.openai.npy"))
        for m in extra_modes:
            p = CACHE / f"llm_repr.{s}.{m}.npy"
            extra_blocks[m].append(np.load(p) if p.exists()
                                   else get_or_compute_llm_repr(s, m, img_s))
        vol_lists.append(idx["VolumeName"].tolist())
    img = np.concatenate(img_blocks, 0)
    sfr = np.concatenate(sfr_blocks, 0)
    har = np.concatenate(har_blocks, 0)
    all_vols = sum(vol_lists, [])
    vol_to_row = {v: i for i, v in enumerate(all_vols)}
    out = {"img": img, "repr_sfr": sfr, "repr_harrier": har, "vol_to_row": vol_to_row}
    if have_openai:
        out["repr_openai"] = np.concatenate(oai_blocks, 0)
    for m in extra_modes:
        out[f"repr_{m}"] = np.concatenate(extra_blocks[m], 0)
    return out


# ---------------------------------------------------------------------------
# Build aligned (img_feats, llm_repr, y_true) per phenotype-manifest split.
# ---------------------------------------------------------------------------
def gather_for_split(manifest_split: pd.DataFrame, labels_df: pd.DataFrame,
                     label_cols: list[str], lookup: dict,
                     *, drop_no_visit: bool = True):
    """Return dict with img/repr_sfr/repr_harrier/y_true for the volumes in
    `manifest_split` that (a) appear in our cached img_feats lookup,
    (b) appear in `labels_df` (so we can read y_true), and
    (c) have a non-null visit_occurrence_id when drop_no_visit=True
        (these are missing-label volumes per the per_ct README, NOT true
        negatives, so they must be excluded from train and eval).
    """
    if drop_no_visit and "visit_occurrence_id" in manifest_split.columns:
        n0 = len(manifest_split)
        manifest_split = manifest_split[manifest_split["visit_occurrence_id"].notna()]
        n1 = len(manifest_split)
        if n0 != n1:
            print(f"    drop_no_visit: {n0} → {n1} volumes ({n0-n1} no-visit dropped)")
    rows = []
    label_idx = labels_df.index
    for _, r in manifest_split.iterrows():
        vol = r["VolumeName"]
        if vol in lookup["vol_to_row"] and vol in label_idx:
            rows.append((vol, lookup["vol_to_row"][vol]))
    if not rows:
        return None
    vols, src_rows = zip(*rows)
    src_rows = np.asarray(src_rows, dtype=np.int64)
    img = lookup["img"][src_rows]
    sfr = lookup["repr_sfr"][src_rows]
    har = lookup["repr_harrier"][src_rows]
    y = labels_df.loc[list(vols)][label_cols].values.astype(np.float32)
    out = {"vols": list(vols), "img": img, "repr_sfr": sfr,
           "repr_harrier": har, "y": y}
    for key in ("repr_openai", "repr_f2llm", "repr_gteqwen2"):
        if key in lookup:
            out[key] = lookup[key][src_rows]
    return out


# ---------------------------------------------------------------------------
# Linear-probe trainer (on llm_repr).
# ---------------------------------------------------------------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def train_cbm(img_train, y_train, img_val, y_val, *, label_names,
              clip_emb_gpu, epochs=200, patience=10, lr=2e-4,
              weight_decay=1e-8, batch_size=512):
    """CBM: forward = (img @ clip_emb.T) @ W. clip_emb_gpu is the on-GPU
    bf16 (M, 768) tensor. Returns (predict_fn, model, best_val_auc, n_epochs).
    """
    M = clip_emb_gpu.shape[0]
    n_labels = y_train.shape[1]
    device = "cuda"
    model = LinearHead(M, n_labels).to(device)
    nn.init.zeros_(model.linear.weight)
    nn.init.zeros_(model.linear.bias)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    tr = DataLoader(TensorDataset(torch.from_numpy(img_train).float(),
                                  torch.from_numpy(y_train).float()),
                    batch_size=batch_size, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(img_val).float(),
                                  torch.from_numpy(y_val).float()),
                    batch_size=batch_size)

    best_val = -np.inf
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x = x.to(device); y = y.to(device)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                sims = x @ clip_emb_gpu.T            # (B, M)
                logits = model(sims.float())
                loss = bce(logits, y)
            loss.backward()
            optim.step()

        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, y in va:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    sims = x.to(device) @ clip_emb_gpu.T
                    logits = model(sims.float())
                ps.append(torch.sigmoid(logits.float()).cpu().numpy())
                ys.append(y.numpy())
        vp = np.concatenate(ps, 0); vy = np.concatenate(ys, 0)
        v_auc = mean_auc(per_label_auc(vp, vy, label_names))
        if v_auc > best_val + 1e-5:
            best_val = v_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    @torch.no_grad()
    def predict(img_x):
        model.eval()
        out = []
        for s in range(0, len(img_x), batch_size):
            xb = torch.from_numpy(img_x[s:s+batch_size]).float().to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                sims = xb @ clip_emb_gpu.T
                logits = model(sims.float())
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out, 0)

    return predict, model, best_val, ep + 1


def cbm_direct_audit(W: np.ndarray, label_names: list[str], concepts: list[str],
                     *, k: int = 100, out_prefix: str,
                     per_label_test_auc: dict | None = None) -> None:
    """Directly use the CBM linear weight as concept importance.

    Each W[label, i] is the contribution of concept i to the logit for that
    label, so it doesn't need any cosine alignment / centering / embedding
    space considerations. This is the cleanest interpretation when the
    bottleneck features are concept_sim themselves.
    """
    rows = []
    payload = {}
    for li, lbl in enumerate(label_names):
        v = W[li]
        pos_idx = np.where(v > 0)[0]
        neg_idx = np.where(v < 0)[0]
        pos_sorted = pos_idx[np.argsort(-v[pos_idx])][:k]
        neg_sorted = neg_idx[np.argsort(v[neg_idx])][:k]
        positive = [{"rank": r + 1, "concept": concepts[ci],
                     "weight": float(v[ci])}
                    for r, ci in enumerate(pos_sorted)]
        negative = [{"rank": r + 1, "concept": concepts[ci],
                     "weight": float(v[ci])}
                    for r, ci in enumerate(neg_sorted)]
        for r, ci in enumerate(pos_sorted, start=1):
            rows.append({"label": lbl, "sign": "positive", "rank": r,
                         "concept": concepts[ci], "weight": float(v[ci])})
        for r, ci in enumerate(neg_sorted, start=1):
            rows.append({"label": lbl, "sign": "negative", "rank": r,
                         "concept": concepts[ci], "weight": float(v[ci])})
        stats = {"n_positive": int(len(pos_idx)), "n_negative": int(len(neg_idx)),
                 "max_weight": float(v.max()), "min_weight": float(v.min())}
        if per_label_test_auc is not None and lbl in per_label_test_auc:
            stats["test_auc"] = per_label_test_auc[lbl]
        payload[lbl] = {"positive": positive, "negative": negative, "stats": stats}
    out_csv = AUDIT_DIR / f"{out_prefix}.csv"
    out_json = AUDIT_DIR / f"{out_prefix}.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  audit → {out_csv}  ({len(label_names)} labels × top-{k})")


def _fit_lbfgs_multilabel(train_x, train_y, *, C, max_iter, balanced):
    """Per-label sklearn LogisticRegression (LBFFS) on z-scored features.

    A linear sigmoid/BCE multi-label head has NO cross-label parameter sharing,
    so fitting each label independently is the same model class — just solved to
    the exact (deterministic, convex) L2-regularized optimum instead of via Adam.
    Weights are returned in RAW feature space (W: n_labels x in_dim, b: n_labels)
    so the saved coefs + cosine-alignment audit are byte-identical in convention
    to the Adam head. joblib-parallel over labels (set OMP_NUM_THREADS=1 to scale).
    """
    from sklearn.linear_model import LogisticRegression
    from joblib import Parallel, delayed

    mu = train_x.mean(0).astype(np.float64)
    sd = train_x.std(0).astype(np.float64) + 1e-8
    Xz = ((train_x.astype(np.float64) - mu) / sd).astype(np.float32)
    n_labels = train_y.shape[1]
    cw = "balanced" if balanced else None

    def _fit(j):
        yj = train_y[:, j].astype(np.int8)
        if yj.min() == yj.max():                       # single-class column
            p = float(np.clip(yj.mean(), 1e-6, 1 - 1e-6))
            return np.zeros(train_x.shape[1], np.float64), float(np.log(p / (1 - p)))
        clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", class_weight=cw)
        clf.fit(Xz, yj)
        return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])

    res = Parallel(n_jobs=-1, backend="loky")(delayed(_fit)(j) for j in range(n_labels))
    Wz = np.stack([r[0] for r in res])                 # (n_labels, in_dim) z-space
    bz = np.array([r[1] for r in res])
    W_raw = Wz / sd[None, :]                            # un-standardize -> raw space
    b_raw = bz - (W_raw * mu[None, :]).sum(1)
    return W_raw.astype(np.float32), b_raw.astype(np.float32)


def train_linear_probe(train_x, train_y, val_x, val_y, *, epochs=200,
                       patience=10, lr=2e-4, weight_decay=1e-8,
                       batch_size=512, label_names=None,
                       solver="adam", lbfgs_C=1.0, lbfgs_max_iter=2000,
                       lbfgs_balanced=False, bootstrap=False, seed=42) -> tuple:
    """Returns (test-call function, model, best_val_auc, n_epochs)."""
    in_dim = train_x.shape[1]
    out_dim = train_y.shape[1]
    device = "cuda"

    if solver == "lbfgs":
        if bootstrap:                                  # resample TRAIN with replacement
            rng = np.random.default_rng(seed)          # (fixed split -> seed needs a data source)
            idx = rng.integers(0, len(train_y), len(train_y))
            train_x = train_x[idx]
            train_y = train_y[idx]
        W_raw, b_raw = _fit_lbfgs_multilabel(
            train_x, train_y, C=lbfgs_C, max_iter=lbfgs_max_iter, balanced=lbfgs_balanced)
        model = LinearHead(in_dim, out_dim).to(device)
        with torch.no_grad():
            model.linear.weight.copy_(torch.from_numpy(W_raw))
            model.linear.bias.copy_(torch.from_numpy(b_raw))

        @torch.no_grad()
        def predict(x):
            model.eval()
            out = []
            for s in range(0, len(x), batch_size):
                xb = torch.from_numpy(x[s:s + batch_size]).float().to(device)
                out.append(torch.sigmoid(model(xb)).cpu().numpy())
            return np.concatenate(out, 0)

        best_val = mean_auc(per_label_auc(predict(val_x), val_y, label_names))
        return predict, model, best_val, 1

    model = LinearHead(in_dim, out_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()

    tr = DataLoader(TensorDataset(torch.from_numpy(train_x).float(),
                                  torch.from_numpy(train_y).float()),
                    batch_size=batch_size, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(val_x).float(),
                                  torch.from_numpy(val_y).float()),
                    batch_size=batch_size)

    best_val = -np.inf
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x = x.to(device); y = y.to(device)
            optim.zero_grad()
            loss = bce(model(x), y)
            loss.backward()
            optim.step()

        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, y in va:
                ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
                ys.append(y.numpy())
        vp = np.concatenate(ps, 0); vy = np.concatenate(ys, 0)
        # Use macro-mean AUC across labels with both classes present
        v_auc = mean_auc(per_label_auc(vp, vy, label_names))
        if v_auc > best_val + 1e-5:
            best_val = v_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    @torch.no_grad()
    def predict(x):
        model.eval()
        out = []
        for s in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[s:s+batch_size]).float().to(device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
        return np.concatenate(out, 0)

    return predict, model, best_val, ep + 1


# ---------------------------------------------------------------------------
# Per-(labelset, split) zero-shot evaluation: plain / sfr / harrier.
# ---------------------------------------------------------------------------
def run_zeroshot_split(name: str, ds: dict, label_names: list[str],
                       label_emb_sfr: dict, label_emb_har: dict,
                       pos_clip: np.ndarray, neg_clip: np.ndarray,
                       prefix: str, summary_rows: list[dict]):
    for method, probs_fn in [
        ("plain", lambda: softmax_pos_neg(ds["img"], pos_clip, neg_clip)),
        ("sfr",   lambda: softmax_pos_neg(ds["repr_sfr"], label_emb_sfr["pos"], label_emb_sfr["neg"])),
        ("harrier", lambda: softmax_pos_neg(ds["repr_harrier"], label_emb_har["pos"], label_emb_har["neg"])),
    ]:
        probs = probs_fn()
        auc = per_label_auc(probs, ds["y"], label_names)
        m = mean_auc(auc)
        save_payload(prefix, name, method, probs, auc, m, len(ds["y"]),
                     label_names, summary_rows)


# LEGACY WARNING: this function normalizes the probe weights W into a cosine
# similarity and does NOT implement manuscript Eq. 2. The authoritative
# alignment score for Figs. 5-6 is the raw-unnormalized-weight implementation
# in analysis/experiments/exp4_confounder_audit/. Kept only because its cosine
# ranking is used as subset-selection input (and only runs when
# SKIP_INLINE_AUDIT != 1).
def cosine_align_audit(W: np.ndarray, label_names: list[str], llm: str,
                       *, k: int = 100, out_prefix: str,
                       per_label_test_auc: dict | None = None,
                       center_concepts: bool = True) -> None:
    """CLEAR-style per-label concept importance via cosine alignment in LLM
    space. Mirrors `examples/concepts/concept_importance.py`:
      - filter to alignments > 0 for "positive" / < 0 for "negative"
      - sort by magnitude within each sign
      - top-K from each (or fewer if not enough exist).

    Sentence-transformer embeddings (SFR / Harrier) are heavily anisotropic
    — they cluster in a narrow cone of the unit sphere. With raw cosine, a
    well-trained probe weight can easily land in a half-space where every
    concept has alignment <= 0 (so n_positive == 0 even though AUC is high).
    Standard fix: subtract the global mean concept embedding before scoring,
    so alignment measures direction *within* the cluster. center_concepts=True
    is on by default; pass False for the strict CLEAR formulation.
    """
    bank = load_concept_bank(llm=llm, device="cpu")
    concepts = bank["concepts"]
    llm_emb = bank["llm_emb"].numpy()  # already L2-normalized
    if center_concepts:
        mean_emb = llm_emb.mean(axis=0, keepdims=True)
        e = llm_emb - mean_emb
        e = e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-9)
    else:
        e = llm_emb
    W_norms = np.linalg.norm(W, axis=1, keepdims=True).clip(1e-9)
    sims = (W / W_norms) @ e.T   # (n_labels, n_concepts)

    rows = []
    payload = {}
    for li, lbl in enumerate(label_names):
        v = sims[li]
        pos_idx = np.where(v > 0)[0]
        neg_idx = np.where(v < 0)[0]
        pos_sorted = pos_idx[np.argsort(-v[pos_idx])][:k]
        neg_sorted = neg_idx[np.argsort(v[neg_idx])][:k]
        positive = [{"rank": r + 1, "concept": concepts[ci],
                     "alignment": float(v[ci])}
                    for r, ci in enumerate(pos_sorted)]
        negative = [{"rank": r + 1, "concept": concepts[ci],
                     "alignment": float(v[ci])}
                    for r, ci in enumerate(neg_sorted)]
        for r, ci in enumerate(pos_sorted, start=1):
            rows.append({"label": lbl, "sign": "positive", "rank": r,
                         "concept": concepts[ci], "alignment": float(v[ci])})
        for r, ci in enumerate(neg_sorted, start=1):
            rows.append({"label": lbl, "sign": "negative", "rank": r,
                         "concept": concepts[ci], "alignment": float(v[ci])})
        stats = {
            "n_positive": int(len(pos_idx)),
            "n_negative": int(len(neg_idx)),
            "max_alignment": float(v.max()),
            "min_alignment": float(v.min()),
        }
        if per_label_test_auc is not None and lbl in per_label_test_auc:
            stats["test_auc"] = per_label_test_auc[lbl]
        payload[lbl] = {"positive": positive, "negative": negative, "stats": stats}
    out_csv = AUDIT_DIR / f"{out_prefix}.csv"
    out_json = AUDIT_DIR / f"{out_prefix}.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  audit → {out_csv}  ({len(label_names)} labels × top-{k})")


def save_payload(prefix, split, method, probs, auc, m, n, labels, summary_rows):
    out = EXTERNAL_DIR / f"{prefix}__{split}__{method}__results.json"
    np.save(EXTERNAL_DIR / f"{prefix}__{split}__{method}__probs.npy", probs)
    payload = {
        "prefix": prefix, "split": split, "method": method,
        "n_volumes": int(n), "n_labels": len(labels),
        "labels": labels, "mean_auc": m, "per_label_auc": auc,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    summary_rows.append({"prefix": prefix, "split": split, "method": method,
                         "n_volumes": int(n), "mean_auc": m})
    print(f"  [{prefix}/{split}/{method}] mean AUC = {m:.4f}  ({len(labels)} labels)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_test_positives", type=int, default=50,
                    help="filter phecodes to those with >= this many positives in TEST")
    ap.add_argument("--labelsets", nargs="*", default=["phenotype", "pe"])
    ap.add_argument("--methods", nargs="*",
                    default=["plain", "sfr", "harrier", "openai",
                             "linear_sfr", "linear_harrier", "linear_openai",
                             "cbm"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-8)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42,
                    help="seeds torch / numpy / random + cuda for reproducibility")
    ap.add_argument("--out_suffix", default="",
                    help="appended to linear_/cbm method tag in filenames "
                         "(e.g. __seed7__lr1e-1) so multi-seed runs don't clobber")
    ap.add_argument("--solver", choices=["adam", "lbfgs"], default="adam",
                    help="linear-probe solver. lbfgs = per-label sklearn LogisticRegression "
                         "(deterministic, z-scored, joblib-parallel); same linear model class "
                         "as the Adam multi-label head, solved to the exact L2 optimum.")
    ap.add_argument("--lbfgs_C", type=float, default=1.0,
                    help="inverse L2 strength for the LBFGS linear probe.")
    ap.add_argument("--lbfgs_max_iter", type=int, default=2000)
    ap.add_argument("--lbfgs_balanced", action="store_true",
                    help="class_weight='balanced' for rare phenotypes.")
    ap.add_argument("--bootstrap", action="store_true",
                    help="resample the TRAIN set with replacement (seeded by --seed) before "
                         "fitting. Use for the 20-seed concept-distribution under LBFGS: the "
                         "fixed train/val split makes a convex LBFGS fit deterministic, so the "
                         "seed must drive data resampling to produce a distribution.")
    args = ap.parse_args()

    import random as _random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    S = args.out_suffix or ""

    # ------------------------------------------------------------------ load
    print("=== Loading phenotype labels + manifest ===")
    labels_df = pd.read_parquet(PHENOTYPE_LABELS)            # 22461 × 1692
    manifest = pd.read_parquet(PHENOTYPE_MANIFEST)            # 22461 × 7
    phen = pd.read_csv(PHENOTYPE_NAMES_CSV, dtype=str)
    phen["phecode"] = phen["phecode"].str.strip()
    phen["phecode_str"] = phen["phecode_str"].str.strip()
    phecode_to_str = dict(zip(phen["phecode"], phen["phecode_str"]))
    print(f"  labels: {labels_df.shape}, manifest: {manifest.shape}")

    # ------------------------------------------------------------------ lookup
    # New ST backends (f2llm / gteqwen2) are projected on the fly from their
    # concept banks; sfr / harrier / openai reuse their cached llm_repr.
    EXTRA_LLM = [m for m in ("f2llm", "gteqwen2")
                 if m in args.methods or f"linear_{m}" in args.methods]
    print("=== Building INSPECT cache lookup ===")
    lookup = build_inspect_lookup(extra_modes=EXTRA_LLM)
    print(f"  {len(lookup['vol_to_row'])} volumes in cache")

    # ----------------------------------------------------------------- splits
    # Test set used for label selection: drop no-visit (missing-label) rows
    # and intersect with our cache.
    test_manifest = manifest[(manifest["split"] == "test") &
                             (manifest["visit_occurrence_id"].notna())]
    test_vols = test_manifest["VolumeName"].values
    test_vols = [v for v in test_vols if v in lookup["vol_to_row"]]
    test_labels = labels_df.loc[test_vols]
    print(f"  test (with-visit, in cache): {len(test_labels)} rows")

    summary_rows: list[dict] = []

    # =========================================================================
    # A) PHENOTYPE labelset
    # =========================================================================
    if "phenotype" in args.labelsets:
        prev = test_labels.sum(axis=0)
        keep_phecodes = sorted(prev[prev >= args.min_test_positives].index.tolist())
        keep_names = [phecode_to_str.get(p, p) for p in keep_phecodes]
        print(f"\n=== Phenotype: {len(keep_phecodes)} phecodes "
              f"(>= {args.min_test_positives} positives in test) ===")
        # Save filtered list
        pd.DataFrame({"phecode": keep_phecodes, "phecode_str": keep_names,
                      "test_pos": prev.loc[keep_phecodes].astype(int).values}
                     ).to_csv(EXTERNAL_DIR / "phenotype__test__filtered_phecodes.csv",
                              index=False)

        # Build (img, repr_sfr, repr_harrier, y_true) per split
        gathered = {}
        for split_name, split_label in [("train", "train"), ("valid", "validation"),
                                        ("test", "test")]:
            ms = manifest[manifest["split"] == split_label]
            sub = gather_for_split(ms, labels_df, keep_phecodes, lookup)
            print(f"  {split_name}: {len(sub['y']) if sub else 0} usable volumes")
            gathered[split_name] = sub

        # Pre-encode label prompts (3D-CLIP, SFR, Harrier)
        print("  encoding label prompts (3D-CLIP)...")
        load_3d_clip()
        pos_clip = encode_texts_clip([n.lower() for n in keep_names])
        neg_clip = encode_texts_clip([f"no {n.lower()}" for n in keep_names])
        free_3d_clip()

        print("  encoding label prompts (SFR)...")
        emb_sfr = get_label_embeddings_for(keep_names, "sfr")
        print("  encoding label prompts (Harrier)...")
        emb_har = get_label_embeddings_for(keep_names, "harrier")
        emb_oai = None
        any_has_oai = any(s is not None and "repr_openai" in s for s in gathered.values())
        if any_has_oai and ("openai" in args.methods or "linear_openai" in args.methods):
            print("  encoding label prompts (OpenAI)...")
            try:
                emb_oai = get_label_embeddings_for(keep_names, "openai")
            except Exception as e:
                print(f"  [skip openai for phenotype labelset] {e}")
                any_has_oai = False  # disable downstream openai usage
        emb_extra = {}
        for m in EXTRA_LLM:
            if m in args.methods:
                print(f"  encoding label prompts ({m})...")
                emb_extra[m] = get_label_embeddings_for(keep_names, m)

        # Zero-shot per split
        for split_name, sub in gathered.items():
            if sub is None:
                continue
            for method in ("plain", "sfr", "harrier", "openai", "f2llm", "gteqwen2"):
                if method not in args.methods:
                    continue
                if method == "openai" and (emb_oai is None or "repr_openai" not in sub):
                    continue
                if method in ("f2llm", "gteqwen2") and (
                        method not in emb_extra or f"repr_{method}" not in sub):
                    continue
                if method == "plain":
                    probs = softmax_pos_neg(sub["img"], pos_clip, neg_clip)
                elif method == "sfr":
                    probs = softmax_pos_neg(sub["repr_sfr"], emb_sfr["pos"], emb_sfr["neg"])
                elif method == "harrier":
                    probs = softmax_pos_neg(sub["repr_harrier"], emb_har["pos"], emb_har["neg"])
                elif method == "openai":
                    probs = softmax_pos_neg(sub["repr_openai"], emb_oai["pos"], emb_oai["neg"])
                else:  # f2llm / gteqwen2
                    eo = emb_extra[method]
                    probs = softmax_pos_neg(sub[f"repr_{method}"], eo["pos"], eo["neg"])
                auc = per_label_auc(probs, sub["y"], keep_names)
                m = mean_auc(auc)
                save_payload("phenotype", split_name, method, probs, auc, m,
                             len(sub["y"]), keep_names, summary_rows)

        # Linear probing per LLM (only run those with cached llm_repr)
        llm_pairs = [("sfr", "repr_sfr"), ("harrier", "repr_harrier")]
        if any_has_oai:
            llm_pairs.append(("openai", "repr_openai"))
        # New ST backends (f2llm / gteqwen2): include when linear_<m> is
        # requested; repr_<m> is loaded into the lookup via EXTRA_LLM above.
        for _m in ("f2llm", "gteqwen2"):
            if f"linear_{_m}" in args.methods:
                llm_pairs.append((_m, f"repr_{_m}"))
        for llm, repr_key in llm_pairs:
            method = f"linear_{llm}"
            if method not in args.methods:
                continue
            if gathered["train"] is None or gathered["test"] is None:
                continue
            print(f"  training linear probe ({llm})...")
            tr = gathered["train"]; va = gathered["valid"]; te = gathered["test"]
            # If no valid set, use 5% of train as val
            if va is None:
                rng = np.random.default_rng(42)
                idx = rng.permutation(len(tr["y"]))
                n_val = max(int(0.05 * len(idx)), 1)
                va_x = tr[repr_key][idx[:n_val]]
                va_y = tr["y"][idx[:n_val]]
                tr_x = tr[repr_key][idx[n_val:]]
                tr_y = tr["y"][idx[n_val:]]
            else:
                tr_x = tr[repr_key]; tr_y = tr["y"]
                va_x = va[repr_key]; va_y = va["y"]
            t0 = time.time()
            predict, model, best_val, n_ep = train_linear_probe(
                tr_x, tr_y, va_x, va_y,
                epochs=args.epochs, patience=args.patience,
                lr=args.lr, weight_decay=args.weight_decay,
                batch_size=args.batch_size, label_names=keep_names,
                solver=args.solver, lbfgs_C=args.lbfgs_C,
                lbfgs_max_iter=args.lbfgs_max_iter, lbfgs_balanced=args.lbfgs_balanced,
                bootstrap=args.bootstrap, seed=args.seed)
            print(f"    {n_ep} epochs in {time.time()-t0:.0f}s; val_auc={best_val:.4f}")

            # Eval all splits with the trained probe
            test_per_label = None
            for split_name, sub in gathered.items():
                if sub is None:
                    continue
                probs = predict(sub[repr_key])
                auc = per_label_auc(probs, sub["y"], keep_names)
                m = mean_auc(auc)
                if split_name == "test":
                    test_per_label = auc
                save_payload("phenotype", split_name, method + S, probs, auc, m,
                             len(sub["y"]), keep_names, summary_rows)

            # Save coefficients for downstream audit
            torch.save({
                "W": model.linear.weight.detach().cpu(),
                "b": model.linear.bias.detach().cpu(),
                "labels": keep_names,
                "phecodes": keep_phecodes,
                "llm": llm,
                "best_val_auc": best_val,
                "test_per_label_auc": test_per_label,
                "seed": int(args.seed),
                "lr": float(args.lr),
            }, EXTERNAL_DIR / f"phenotype__linear_{llm}{S}__coefs.pt")

            # Per-label concept importance via cosine alignment in LLM space.
            # Skippable: the 20-seed sweep/seeds don't need this per-call audit
            # (concept_audit_20seeds.py recomputes it from the saved coefs), so
            # SKIP_INLINE_AUDIT=1 avoids reloading the multi-GB concept bank on
            # every LR/seed and keeps outputs/v1/audit/ clean.
            if os.environ.get("SKIP_INLINE_AUDIT") != "1":
                cosine_align_audit(
                    W=model.linear.weight.detach().cpu().numpy(),
                    label_names=keep_names, llm=llm, k=100,
                    out_prefix=f"phenotype__linear_{llm}{S}__concept_importance",
                    per_label_test_auc=test_per_label,
                )

        # ----------------------------------------------------------------
        # CBM (linear over the 376k-d concept_sim) — gives per-concept
        # weights directly, no embedding-cone effects.
        # ----------------------------------------------------------------
        if "cbm" in args.methods:
            print(f"\n  training CBM (376k-concept linear head)...")
            bank_full = load_concept_bank(llm="sfr", device="cpu")  # only need clip_text_emb
            clip_emb_gpu = bank_full["clip_text_emb"].cuda().to(torch.bfloat16)
            concepts = bank_full["concepts"]
            tr = gathered["train"]; va = gathered["valid"]; te = gathered["test"]
            t0 = time.time()
            predict, model, best_val, n_ep = train_cbm(
                tr["img"], tr["y"], va["img"], va["y"],
                label_names=keep_names, clip_emb_gpu=clip_emb_gpu,
                epochs=args.epochs, patience=args.patience,
                lr=args.lr, weight_decay=args.weight_decay,
                batch_size=args.batch_size)
            print(f"    {n_ep} epochs in {time.time()-t0:.0f}s; val_auc={best_val:.4f}")

            test_per_label = None
            for split_name, sub in gathered.items():
                if sub is None:
                    continue
                probs = predict(sub["img"])
                auc = per_label_auc(probs, sub["y"], keep_names)
                m = mean_auc(auc)
                if split_name == "test":
                    test_per_label = auc
                save_payload("phenotype", split_name, f"cbm{S}", probs, auc, m,
                             len(sub["y"]), keep_names, summary_rows)

            torch.save({
                "W": model.linear.weight.detach().cpu(),
                "b": model.linear.bias.detach().cpu(),
                "labels": keep_names,
                "phecodes": keep_phecodes,
                "concepts": concepts,
                "best_val_auc": best_val,
                "test_per_label_auc": test_per_label,
                "seed": int(args.seed),
                "lr": float(args.lr),
            }, EXTERNAL_DIR / f"phenotype__cbm{S}__coefs.pt")

            cbm_direct_audit(
                W=model.linear.weight.detach().cpu().numpy(),
                label_names=keep_names, concepts=concepts, k=100,
                out_prefix=f"phenotype__cbm{S}__concept_importance",
                per_label_test_auc=test_per_label,
            )
            del clip_emb_gpu, bank_full
            torch.cuda.empty_cache()

    # =========================================================================
    # B) PE labelset (3 native INSPECT tasks)
    # =========================================================================
    if "pe" in args.labelsets:
        from clear3d.paths import LABELS_PE
        print(f"\n=== PE labelset ({len(LABELS_PE)} labels) ===")
        # PE labels live in the INSPECT pe_labels CSVs and are joined via
        # `clear3d.data.load_pe_split` per inspect_* split. Concatenate them.
        pe_per_inspect_split = {}
        for s in ("inspect_train", "inspect_valid", "inspect_test"):
            df = load_pe_split(s)
            pe_per_inspect_split[s] = df.set_index("VolumeName")[LABELS_PE]
        pe_labels_df = pd.concat(list(pe_per_inspect_split.values()))
        pe_labels_df = pe_labels_df[~pe_labels_df.index.duplicated(keep="first")]

        gathered = {}
        for split_name, split_label in [("train", "train"), ("valid", "validation"),
                                        ("test", "test")]:
            ms = manifest[manifest["split"] == split_label]
            sub = gather_for_split(ms, pe_labels_df, LABELS_PE, lookup)
            print(f"  {split_name}: {len(sub['y']) if sub else 0} usable volumes")
            gathered[split_name] = sub

        # Encode prompts
        print("  encoding label prompts (3D-CLIP)...")
        load_3d_clip()
        pe_pretty = ["pulmonary embolism", "acute pulmonary embolism",
                     "subsegmental pulmonary embolism"]
        pos_clip = encode_texts_clip(pe_pretty)
        neg_clip = encode_texts_clip([f"no {p}" for p in pe_pretty])
        free_3d_clip()

        emb_sfr = get_label_embeddings_for(pe_pretty, "sfr")
        emb_har = get_label_embeddings_for(pe_pretty, "harrier")
        emb_oai_pe = None
        any_has_oai_pe = any(s is not None and "repr_openai" in s for s in gathered.values())
        if any_has_oai_pe and ("openai" in args.methods or "linear_openai" in args.methods):
            try:
                emb_oai_pe = get_label_embeddings_for(pe_pretty, "openai")
            except Exception as e:
                print(f"  [skip openai for PE labelset] {e}")
                any_has_oai_pe = False

        # Zero-shot
        for split_name, sub in gathered.items():
            if sub is None:
                continue
            for method in ("plain", "sfr", "harrier", "openai"):
                if method not in args.methods:
                    continue
                if method == "openai" and (emb_oai_pe is None or "repr_openai" not in sub):
                    continue
                if method == "plain":
                    probs = softmax_pos_neg(sub["img"], pos_clip, neg_clip)
                elif method == "sfr":
                    probs = softmax_pos_neg(sub["repr_sfr"], emb_sfr["pos"], emb_sfr["neg"])
                elif method == "harrier":
                    probs = softmax_pos_neg(sub["repr_harrier"], emb_har["pos"], emb_har["neg"])
                else:  # openai
                    probs = softmax_pos_neg(sub["repr_openai"], emb_oai_pe["pos"], emb_oai_pe["neg"])
                auc = per_label_auc(probs, sub["y"], LABELS_PE)
                m = mean_auc(auc)
                save_payload("pe_pheno", split_name, method, probs, auc, m,
                             len(sub["y"]), LABELS_PE, summary_rows)

        # Linear probe per LLM (only the ones with cached repr)
        pe_llm_pairs = [("sfr", "repr_sfr"), ("harrier", "repr_harrier")]
        if any_has_oai_pe:
            pe_llm_pairs.append(("openai", "repr_openai"))
        for llm, repr_key in pe_llm_pairs:
            method = f"linear_{llm}"
            if method not in args.methods:
                continue
            tr = gathered["train"]; va = gathered["valid"]; te = gathered["test"]
            if tr is None:
                continue
            print(f"  training PE linear probe ({llm})...")
            if va is None:
                rng = np.random.default_rng(42)
                idx = rng.permutation(len(tr["y"]))
                n_val = max(int(0.05 * len(idx)), 1)
                va_x = tr[repr_key][idx[:n_val]]
                va_y = tr["y"][idx[:n_val]]
                tr_x = tr[repr_key][idx[n_val:]]
                tr_y = tr["y"][idx[n_val:]]
            else:
                tr_x = tr[repr_key]; tr_y = tr["y"]
                va_x = va[repr_key]; va_y = va["y"]
            t0 = time.time()
            predict, model, best_val, n_ep = train_linear_probe(
                tr_x, tr_y, va_x, va_y,
                epochs=args.epochs, patience=args.patience,
                lr=args.lr, weight_decay=args.weight_decay,
                batch_size=args.batch_size, label_names=LABELS_PE,
                solver=args.solver, lbfgs_C=args.lbfgs_C,
                lbfgs_max_iter=args.lbfgs_max_iter, lbfgs_balanced=args.lbfgs_balanced,
                bootstrap=args.bootstrap, seed=args.seed)
            print(f"    {n_ep} epochs in {time.time()-t0:.0f}s; val_auc={best_val:.4f}")
            test_per_label = None
            for split_name, sub in gathered.items():
                if sub is None:
                    continue
                probs = predict(sub[repr_key])
                auc = per_label_auc(probs, sub["y"], LABELS_PE)
                m = mean_auc(auc)
                if split_name == "test":
                    test_per_label = auc
                save_payload("pe_pheno", split_name, method + S, probs, auc, m,
                             len(sub["y"]), LABELS_PE, summary_rows)
            torch.save({
                "W": model.linear.weight.detach().cpu(),
                "b": model.linear.bias.detach().cpu(),
                "labels": LABELS_PE,
                "llm": llm,
                "best_val_auc": best_val,
                "test_per_label_auc": test_per_label,
                "seed": int(args.seed),
                "lr": float(args.lr),
            }, EXTERNAL_DIR / f"pe_pheno__linear_{llm}{S}__coefs.pt")

            # Per-label concept importance for the 3 PE labels
            cosine_align_audit(
                W=model.linear.weight.detach().cpu().numpy(),
                label_names=LABELS_PE, llm=llm, k=100,
                out_prefix=f"pe_pheno__linear_{llm}{S}__concept_importance",
                per_label_test_auc=test_per_label,
            )

    # ------------------------------------------------------------------ summary
    df = pd.DataFrame(summary_rows)
    out = EXTERNAL_DIR / f"phenotype__summary{S}.csv"
    df.to_csv(out, index=False)
    print(f"\n=== Summary → {out} ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

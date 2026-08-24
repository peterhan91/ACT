"""AUC + per-label results helpers (mirrors eval_all.py's output schema)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def per_label_auc(probs: np.ndarray, y_true: np.ndarray, labels: list[str]) -> dict:
    """Returns {label: auc} skipping any label where y_true is all-zero or all-one."""
    out = {}
    for i, lbl in enumerate(labels):
        y = y_true[:, i]
        if len(np.unique(y)) < 2:
            out[lbl] = float("nan")
        else:
            out[lbl] = float(roc_auc_score(y, probs[:, i]))
    return out


def mean_auc(per_label: dict) -> float:
    vals = [v for v in per_label.values() if v == v]  # drop NaN
    return float(np.mean(vals)) if vals else float("nan")


def softmax_pos_neg(img_feats: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """For each (image, label), P = exp(sim+) / (exp(sim+) + exp(sim-))."""
    img = img_feats / np.linalg.norm(img_feats, axis=-1, keepdims=True).clip(1e-9)
    pos = pos / np.linalg.norm(pos, axis=-1, keepdims=True).clip(1e-9)
    neg = neg / np.linalg.norm(neg, axis=-1, keepdims=True).clip(1e-9)
    sp = img @ pos.T
    sn = img @ neg.T
    # Numerically stable softmax over the 2-element axis.
    m = np.maximum(sp, sn)
    ep = np.exp(sp - m)
    en = np.exp(sn - m)
    return ep / (ep + en)

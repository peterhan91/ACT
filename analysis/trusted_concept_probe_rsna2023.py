#!/usr/bin/env python3
"""LBFGS trusted-concept probes for the 9 RSNA-2023 abdominal-trauma labels.

Mirrors `trusted_concept_probe.py` (which targets the INSPECT 221 phenotypes)
but with two task-specific adaptations:

  * Data gathering uses `load_trauma_split` (row-aligned, no VolumeName
    remapping; rsna2023 h5_idx == labels CSV row index).
  * Solver is **fixed** to sklearn LBFGS with C=1e6, balanced=False,
    max_iter=5000 — the optimal setting from INSPECT.

For each label and each method in {original_topk, trusted, direct}:
  feature = L2_normalize( (img_norm @ clip_emb[idx].T) @ openai_emb[idx] )
            # (N, 3072) — OpenAI-bottlenecked CT embedding through the
            # label's chosen concept set
  probe   = LogisticRegression(C=1e6, max_iter=5000) on z-scored features

Writes the same schema as `trusted_concept_probe.py` so downstream tools
(rank summaries, top-coefficients audit) work without changes:

  outputs/v1/external/rsna2023_trusted_concept_probe_openai_lbfgs/
    trusted_concept_probe_summary.csv
    trusted_concept_probe_overall.csv
    trusted_concept_probe_top_coefficients.csv
    {train,valid,test}__{method}__probs.npz
    manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

# Pin v1 backbone before any clear3d import.
os.environ.setdefault("CLIP3D_RUN_TAG", "v1")
os.environ.setdefault("CLIP3D_BEST", "clip_3d_ctrate_merlin_v1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from clear3d.data import load_trauma_split
from clear3d.paths import CACHE, EXTERNAL_DIR, LABELS_RSNA2023_TRAUMA, ROOT
from trusted_concept_space import slug
from trusted_concept_space_rsna2023 import RSNA2023_LABELS


# ---------------------------------------------------------------------------
# Helpers — kept inline so the module is self-contained.
# ---------------------------------------------------------------------------

def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.float32)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(np.float32)
    if y.sum() <= 0:
        return float("nan")
    return float(average_precision_score(y, p))


def safe_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y.astype(np.float32), p))


def finite_mean(vals) -> float:
    arr = [v for v in vals if v == v and math.isfinite(v)]
    return float(np.mean(arr)) if arr else float("nan")


@dataclass
class ConceptSet:
    method: str
    indices: list[int]
    concepts: list[str]
    tiers: list[str]
    scores: list[float]


class IndexedConceptBanks:
    def __init__(self, *, openai_path: Path, clip_path: Path):
        self.paths = {"openai": openai_path, "clip": clip_path}
        self._npz = {k: np.load(v, allow_pickle=True) for k, v in self.paths.items()}
        oai_c = [str(c) for c in self._npz["openai"]["concepts"]]
        clip_c = [str(c) for c in self._npz["clip"]["concepts"]]
        if oai_c != clip_c:
            raise ValueError("Concept order mismatch between OpenAI and CLIP banks.")
        self.concepts = oai_c
        self.concept_to_idx = {c: i for i, c in enumerate(self.concepts)}
        self._emb: dict[str, np.ndarray] = {}

    def emb(self, space: str) -> np.ndarray:
        if space not in self._emb:
            self._emb[space] = self._npz[space]["emb"].astype(np.float32, copy=False)
        return self._emb[space]

    def select(self, indices: list[int], space: str) -> np.ndarray:
        if not indices:
            raise ValueError("cannot select zero concept indices")
        e = self.emb(space)[np.asarray(indices, dtype=np.int64)].astype(np.float32, copy=True)
        return norm_rows(e)


def openai_projection(img_norm: np.ndarray, clip_sel: np.ndarray, openai_sel: np.ndarray) -> np.ndarray:
    """proj = L2norm( (img @ clip_emb[idx].T) @ openai_emb[idx] ).

    Same as clear3d.projection.project_to_llm restricted to a selected concept
    set. Matches `trusted_concept_probe.py:openai_concept_projection`.
    """
    sims = img_norm @ clip_sel.T
    proj = (sims @ openai_sel).astype(np.float32, copy=False)
    return norm_rows(proj)


def standardize_features(xt: np.ndarray, *more: np.ndarray, eps: float = 1e-8):
    mu = xt.mean(0, keepdims=True)
    sd = xt.std(0, keepdims=True) + eps
    return tuple((arr - mu) / sd for arr in (xt, *more))


def train_lbfgs(x_tr, y_tr, x_va, y_va, *, C: float, max_iter: int):
    m = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter)
    m.fit(x_tr, y_tr.astype(int))
    va_p = m.predict_proba(x_va)[:, 1]
    return m, float(safe_auc(y_va, va_p)), int(m.n_iter_[0])


def predict_lbfgs(m, x) -> np.ndarray:
    return m.predict_proba(x)[:, 1].astype(np.float32)


# ---------------------------------------------------------------------------
# Concept-set builders — one per --method choice.
# ---------------------------------------------------------------------------

def load_audit(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_trusted_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def unique_keep_order(rows: list[dict], max_count: int) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        i = int(r["concept_index"])
        if i in seen:
            continue
        seen.add(i)
        out.append(r)
        if max_count > 0 and len(out) >= max_count:
            break
    return out


def original_topk_set(label: str, audit: dict, bank: IndexedConceptBanks, k: int) -> ConceptSet:
    rows = audit.get(label, {}).get("positive", [])[:k]
    idxs, names, scores = [], [], []
    for r in rows:
        c = str(r["concept"])
        i = bank.concept_to_idx.get(c)
        if i is None:
            continue
        idxs.append(i); names.append(c)
        scores.append(float(r.get("alignment", r.get("mean", 0.0))))
    return ConceptSet("original_topk", idxs, names, ["original"] * len(idxs), scores)


def trusted_set(label: str, trusted_dir: Path, *, max_count: int, direct_only: bool) -> ConceptSet:
    csv_path = trusted_dir / f"{slug(label)}.trusted_concepts.csv"
    rows = load_trusted_rows(csv_path)
    if direct_only:
        rows = [r for r in rows if r.get("tier") == "direct"]
    rows.sort(key=lambda r: (-float(r.get("score", 0.0)), r.get("tier", ""), r.get("concept", "")))
    rows = unique_keep_order(rows, max_count)
    method = "direct" if direct_only else "trusted"
    return ConceptSet(
        method,
        [int(r["concept_index"]) for r in rows],
        [str(r["concept"]) for r in rows],
        [str(r.get("tier", method)) for r in rows],
        [float(r.get("score", 0.0)) for r in rows],
    )


def coefficient_rows(*, label: str, method: str, model, cset: ConceptSet, top_k: int) -> list[dict]:
    # On the openai feature path the linear weights live in the 3072-d OpenAI
    # space, not 1:1 with concepts. We compute per-concept importance via
    # importance(c) = openai_emb[c] @ (w_z / sd) — same recipe as
    # _audit_trusted_openai_probes.py — but only if the caller passes back
    # the (mu, sd, openai_sel) bundle. For the summary CSV we still surface
    # the top-K |coef| in OpenAI-dim space as a sanity check.
    coef = np.asarray(model.coef_).reshape(-1)
    order = np.argsort(-np.abs(coef))[: min(top_k, len(coef))]
    return [
        {
            "label": label, "method": method, "rank": r + 1,
            "openai_dim": int(j), "coef": float(coef[j]), "abs_coef": float(abs(coef[j])),
        }
        for r, j in enumerate(order)
    ]


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    default_trusted_dir = ROOT / "outputs/v1/audit/trusted_concept_spaces_rsna2023"
    default_audit_json = ROOT / "outputs/v1/audit/rsna2023__linear_openai__concept_importance.json"
    default_out_dir = EXTERNAL_DIR / "rsna2023_trusted_concept_probe_openai_lbfgs"
    ap = argparse.ArgumentParser()
    ap.add_argument("--openai_concept_bank", default=str(ROOT / "concept_bank.openai_emb.npz"))
    ap.add_argument("--clip_concept_bank", default=str(ROOT / "concept_bank.clip_text_emb.v1.npz"))
    ap.add_argument("--trusted_dir", default=str(default_trusted_dir))
    ap.add_argument("--audit_json", default=str(default_audit_json))
    ap.add_argument("--out_dir", default=str(default_out_dir))
    ap.add_argument(
        "--labels", nargs="*", default=None,
        help="Subset of the 9 trauma labels; default is all.",
    )
    ap.add_argument(
        "--methods", nargs="*",
        default=["original_topk", "trusted", "direct"],
        choices=["original_topk", "trusted", "direct"],
    )
    ap.add_argument("--feature_space", default="openai", choices=["openai"],
                    help="Only openai-projected features are supported for the RSNA-2023 track.")
    ap.add_argument("--original_top_k", type=int, default=100)
    ap.add_argument("--max_concepts", type=int, default=512,
                    help="Cap trusted/direct concepts per label; 0 keeps all.")
    ap.add_argument("--lbfgs_C", type=float, default=1e6)
    ap.add_argument("--lbfgs_max_iter", type=int, default=5000)
    ap.add_argument("--top_coef_k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_split", default="rsna2023_train")
    ap.add_argument("--valid_split", default="rsna2023_valid")
    ap.add_argument("--test_split", default="rsna2023_test")
    return ap.parse_args()


def load_split_features(split: str, label_cols: list[str]) -> dict:
    """Returns {img: (N, 768) normalized, y: (N, n_labels), idx_df: DataFrame}."""
    df = load_trauma_split(split)
    img = np.load(CACHE / f"img_feats.{split}.npy")
    if img.shape[0] != len(df):
        n = min(img.shape[0], len(df))
        df = df.head(n).copy()
        img = img[:n]
    return {
        "df": df,
        "img": norm_rows(img),
        "y": df[label_cols].astype(np.float32).values,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trusted_dir = Path(args.trusted_dir)
    audit = load_audit(Path(args.audit_json))

    labels = args.labels if args.labels is not None else list(RSNA2023_LABELS)
    unknown = [l for l in labels if l not in LABELS_RSNA2023_TRAUMA]
    if unknown:
        raise SystemExit(f"Unknown RSNA-2023 labels: {unknown}; choose from {LABELS_RSNA2023_TRAUMA}")
    label_to_col = {l: i for i, l in enumerate(LABELS_RSNA2023_TRAUMA)}

    print("=== Loading RSNA-2023 splits + cached img_feats (v1 backbone) ===")
    gathered = {
        "train": load_split_features(args.train_split, LABELS_RSNA2023_TRAUMA),
        "valid": load_split_features(args.valid_split, LABELS_RSNA2023_TRAUMA),
        "test": load_split_features(args.test_split, LABELS_RSNA2023_TRAUMA),
    }
    for k, g in gathered.items():
        print(f"  {k}: {len(g['y'])} volumes")

    print("=== Loading concept banks (v1 clip_text + openai) ===")
    bank = IndexedConceptBanks(
        openai_path=Path(args.openai_concept_bank),
        clip_path=Path(args.clip_concept_bank),
    )

    result_rows: list[dict] = []
    top_coef_rows: list[dict] = []
    method_tags = [f"{args.feature_space}_{m}" for m in args.methods]
    prob_arrays = {
        tag: {
            split: np.full((len(gathered[split]["y"]), len(labels)), np.nan, dtype=np.float32)
            for split in gathered
        }
        for tag in method_tags
    }
    label_pos = {l: i for i, l in enumerate(labels)}

    for li, label in enumerate(labels):
        col = label_to_col[label]
        y_tr = gathered["train"]["y"][:, col]
        y_va = gathered["valid"]["y"][:, col]
        y_te = gathered["test"]["y"][:, col]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            print(f"[skip] {label}: train/test has one class")
            continue

        candidates: list[ConceptSet] = []
        if "original_topk" in args.methods:
            candidates.append(original_topk_set(label, audit, bank, args.original_top_k))
        if "trusted" in args.methods:
            candidates.append(trusted_set(label, trusted_dir, max_count=args.max_concepts, direct_only=False))
        if "direct" in args.methods:
            candidates.append(trusted_set(label, trusted_dir, max_count=args.max_concepts, direct_only=True))

        for cset in candidates:
            tag = f"{args.feature_space}_{cset.method}"
            if not cset.indices:
                result_rows.append({
                    "label": label, "method": tag, "basis": cset.method,
                    "n_concepts": 0, "status": "skipped_no_concepts",
                })
                print(f"[{li+1:>2}/{len(labels)}] {label:<22s} / {tag}: 0 concepts — SKIP")
                continue

            clip_sel = bank.select(cset.indices, "clip")
            openai_sel = bank.select(cset.indices, "openai")

            xtr = openai_projection(gathered["train"]["img"], clip_sel, openai_sel)
            xva = openai_projection(gathered["valid"]["img"], clip_sel, openai_sel)
            xte = openai_projection(gathered["test"]["img"], clip_sel, openai_sel)
            xtr, xva, xte = standardize_features(xtr, xva, xte)

            t0 = time.time()
            model, best_val, n_iter = train_lbfgs(
                xtr, y_tr, xva, y_va,
                C=args.lbfgs_C, max_iter=args.lbfgs_max_iter,
            )
            ptr = predict_lbfgs(model, xtr)
            pva = predict_lbfgs(model, xva)
            pte = predict_lbfgs(model, xte)
            for split, probs in [("train", ptr), ("valid", pva), ("test", pte)]:
                prob_arrays[tag][split][:, li] = probs

            row = {
                "label": label, "method": tag, "basis": cset.method,
                "n_concepts": len(cset.indices),
                "n_direct": int(sum(t == "direct" for t in cset.tiers)),
                "n_associated": int(sum(t == "associated" for t in cset.tiers)),
                "n_original": int(sum(t == "original" for t in cset.tiers)),
                "train_auc": safe_auc(y_tr, ptr),
                "valid_auc": safe_auc(y_va, pva),
                "test_auc": safe_auc(y_te, pte),
                "test_ap": safe_ap(y_te, pte),
                "test_brier": safe_brier(y_te, pte),
                "n_iter": n_iter,
                "seconds": round(time.time() - t0, 2),
                "status": "ok",
            }
            result_rows.append(row)
            top_coef_rows.extend(coefficient_rows(
                label=label, method=tag, model=model, cset=cset, top_k=args.top_coef_k,
            ))
            print(
                f"[{li+1:>2}/{len(labels)}] {label:<22s} / {tag:<22s}  "
                f"K={len(cset.indices):>4d}  test_auc={row['test_auc']:.4f}  "
                f"({row['seconds']}s)"
            )

    summary = pd.DataFrame(result_rows)
    summary_path = out_dir / "trusted_concept_probe_summary.csv"
    overall_path = out_dir / "trusted_concept_probe_overall.csv"
    top_coef_path = out_dir / "trusted_concept_probe_top_coefficients.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(top_coef_rows).to_csv(top_coef_path, index=False)

    ok = summary[summary["status"] == "ok"].copy() if not summary.empty else summary
    overall = []
    if not ok.empty:
        for tag, df_m in ok.groupby("method"):
            overall.append({
                "method": tag,
                "n_labels": int(len(df_m)),
                "mean_test_auc": finite_mean(df_m["test_auc"]),
                "mean_test_ap": finite_mean(df_m["test_ap"]),
                "mean_test_brier": finite_mean(df_m["test_brier"]),
                "median_n_concepts": float(df_m["n_concepts"].median()),
            })
    pd.DataFrame(overall).to_csv(overall_path, index=False)

    for tag, splits in prob_arrays.items():
        for split, arr in splits.items():
            np.savez_compressed(
                out_dir / f"{split}__{tag}__probs.npz",
                probs=arr,
                labels=np.asarray(labels, dtype=object),
            )

    manifest = {
        "task": "rsna2023_trauma_lbfgs_trusted",
        "tag": "v1",
        "backbone": os.environ.get("CLIP3D_BEST", "(unset)"),
        "openai_concept_bank": str(Path(args.openai_concept_bank)),
        "clip_concept_bank": str(Path(args.clip_concept_bank)),
        "trusted_dir": str(trusted_dir),
        "audit_json": str(Path(args.audit_json)),
        "methods": args.methods,
        "feature_space": args.feature_space,
        "labels": labels,
        "lbfgs_C": args.lbfgs_C,
        "lbfgs_max_iter": args.lbfgs_max_iter,
        "original_top_k": args.original_top_k,
        "max_concepts": args.max_concepts,
        "seed": args.seed,
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {summary_path}")
    print(f"Wrote {overall_path}")
    print(f"Wrote {top_coef_path}")
    if overall:
        print()
        print(pd.DataFrame(overall).to_string(index=False))


if __name__ == "__main__":
    main()

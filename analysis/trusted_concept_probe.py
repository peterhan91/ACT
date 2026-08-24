#!/usr/bin/env python3
"""Per-label probes over original, trusted, and direct OpenAI concept bases.

This script tests whether the radiologist-refined concept spaces retain
phenotype prediction performance. It keeps the global `concept_index` from the
trusted-space CSVs and uses that same index to slice both aligned banks:

    clip features   = cached 3D-CLIP image features @ clip_text_emb[index].T
    openai features = cached OpenAI-space image repr @ openai_emb[index].T

The default run tag is v1 because the trusted spaces were generated under
`outputs/v1/audit/trusted_concept_spaces`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _preconfigure_run_tag() -> str:
    tag = "v1"
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--tag" and i + 1 < len(argv):
            tag = argv[i + 1]
        elif arg.startswith("--tag="):
            tag = arg.split("=", 1)[1]
    os.environ.setdefault("CLIP3D_RUN_TAG", tag)
    return os.environ["CLIP3D_RUN_TAG"]


RUN_TAG = _preconfigure_run_tag()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import warnings as _warnings
_warnings.filterwarnings("ignore", category=ConvergenceWarning)
_warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from clear3d.paths import (
    CACHE,
    EXTERNAL_DIR,
    PHENOTYPE_LABELS,
    PHENOTYPE_MANIFEST,
    PHENOTYPE_NAMES_CSV,
    ROOT,
)
from trusted_concept_space import slug


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-8, None)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = y_true.astype(np.float32)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, y_prob))


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = y_true.astype(np.float32)
    if y.sum() <= 0:
        return float("nan")
    return float(average_precision_score(y, y_prob))


def safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true.astype(np.float32), y_prob))


def finite_mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def build_inspect_lookup() -> dict:
    """Build VolumeName-indexed cached 3D-CLIP image-feature lookup.

    OpenAI-space features are recomputed on the fly per concept selection
    (img_feats @ clip_text_emb[idx].T @ openai_emb[idx]), so the cached
    full-bank llm_repr.openai.npy is no longer needed here.
    """
    splits = ["inspect_train", "inspect_valid", "inspect_test"]
    img_blocks, vol_lists = [], []
    for split in splits:
        idx = pd.read_csv(CACHE / f"volume_index.{split}.csv")
        img_blocks.append(np.load(CACHE / f"img_feats.{split}.npy"))
        vol_lists.append(idx["VolumeName"].tolist())
    all_vols = sum(vol_lists, [])
    return {
        "img": np.concatenate(img_blocks, axis=0),
        "vol_to_row": {v: i for i, v in enumerate(all_vols)},
    }


def gather_for_split(
    manifest_split: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_cols: list[str],
    lookup: dict,
    *,
    drop_no_visit: bool = True,
) -> dict | None:
    if drop_no_visit and "visit_occurrence_id" in manifest_split.columns:
        manifest_split = manifest_split[manifest_split["visit_occurrence_id"].notna()]
    rows = []
    label_idx = labels_df.index
    for _, row in manifest_split.iterrows():
        vol = row["VolumeName"]
        if vol in lookup["vol_to_row"] and vol in label_idx:
            rows.append((vol, lookup["vol_to_row"][vol]))
    if not rows:
        return None
    vols, src_rows = zip(*rows)
    src_rows = np.asarray(src_rows, dtype=np.int64)
    return {
        "vols": list(vols),
        "img": lookup["img"][src_rows],
        "y": labels_df.loc[list(vols)][label_cols].values.astype(np.float32),
    }


class IndexedConceptBanks:
    def __init__(self, paths: dict[str, Path]):
        """`paths` maps feature-space name -> concept-bank npz.

        Always includes 'clip' (used to compute concept-similarity scores) plus
        one or more LLM spaces ('openai', 'f2llm', ...). All banks must share the
        same concept order so a global `concept_index` is reusable across them.
        """
        self.paths = dict(paths)
        self._npz = {name: np.load(path, allow_pickle=True) for name, path in self.paths.items()}
        ref_name = "clip" if "clip" in self._npz else next(iter(self._npz))
        ref_concepts = [str(c) for c in self._npz[ref_name]["concepts"]]
        for name, z in self._npz.items():
            if [str(c) for c in z["concepts"]] != ref_concepts:
                raise ValueError(
                    f"Concept order mismatch between {ref_name} and {name} concept banks; "
                    "cannot safely reuse concept_index across banks."
                )
        self.concepts = ref_concepts
        self.concept_to_idx = {c: i for i, c in enumerate(self.concepts)}
        self._emb: dict[str, np.ndarray] = {}

    def emb(self, feature_space: str) -> np.ndarray:
        if feature_space not in self._emb:
            print(f"Loading {feature_space} concept embeddings: {self.paths[feature_space]}")
            self._emb[feature_space] = self._npz[feature_space]["emb"].astype(np.float32, copy=False)
        return self._emb[feature_space]

    def select_embeddings(self, indices: list[int], feature_space: str) -> np.ndarray:
        if not indices:
            raise ValueError("cannot select zero concept embeddings")
        e = self.emb(feature_space)[np.asarray(indices, dtype=np.int64)].astype(np.float32, copy=True)
        return normalize_rows(e)


@dataclass
class ConceptSet:
    method: str
    indices: list[int]
    concepts: list[str]
    tiers: list[str]
    scores: list[float]


class BinaryLinear(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_binary_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> tuple[BinaryLinear, float, int]:
    model = BinaryLinear(x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()),
        batch_size=batch_size,
        shuffle=True,
    )

    x_valid_t = torch.from_numpy(x_valid).float().to(device)
    y_valid_t = torch.from_numpy(y_valid).float().to(device)
    best_score = -np.inf
    best_state = None
    bad = 0
    n_done = 0

    for epoch in range(epochs):
        n_done = epoch + 1
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(x_valid_t)
            valid_loss = float(loss_fn(logits, y_valid_t).detach().cpu())
            valid_prob = torch.sigmoid(logits).detach().cpu().numpy()
        valid_auc = safe_auc(y_valid, valid_prob)
        score = valid_auc if math.isfinite(valid_auc) else -valid_loss
        if score > best_score + 1e-6:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_score), n_done


@torch.no_grad()
def predict_binary(
    model: BinaryLinear,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).float().to(device)
        out.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def standardize_features(xt: np.ndarray, *more: np.ndarray, eps: float = 1e-8):
    """Z-score per-feature using train stats. Returns z-scored arrays in input order."""
    mu = xt.mean(0, keepdims=True)
    sd = xt.std(0, keepdims=True) + eps
    return tuple((arr - mu) / sd for arr in (xt, *more))


def train_lbfgs_probe(
    x_train: np.ndarray, y_train: np.ndarray,
    x_valid: np.ndarray, y_valid: np.ndarray,
    *, C: float, balanced: bool, max_iter: int,
) -> tuple[LogisticRegression, float, int]:
    model = LogisticRegression(
        C=C, solver="lbfgs", max_iter=max_iter,
        class_weight="balanced" if balanced else None,
    )
    model.fit(x_train, y_train.astype(int))
    valid_probs = model.predict_proba(x_valid)[:, 1]
    best_val = safe_auc(y_valid, valid_probs)
    return model, float(best_val), int(model.n_iter_[0])


def predict_lbfgs(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1].astype(np.float32)


def clip_concept_scores(img_norm: np.ndarray, clip_emb_sel: np.ndarray) -> np.ndarray:
    """CLIP-space concept scores: img_norm @ clip_emb_sel.T  → (N, K)."""
    return (img_norm @ clip_emb_sel.T).astype(np.float32, copy=False)


def llm_concept_projection(
    img_norm: np.ndarray, clip_emb_sel: np.ndarray, llm_emb_sel: np.ndarray,
) -> np.ndarray:
    """Concept-bottlenecked LLM-space CT embedding via the selected concepts.

    Mirrors `clear3d/projection.project_to_llm` but restricted to one label's
    selected concept indices, for any LLM space (openai 3072-d, f2llm 5120-d, ...):
        out = (img @ clip_emb[idx].T) @ llm_emb[idx]   # (N, D_llm)
        out = L2-normalize(out)

    At the full bank, this reproduces the cached `llm_repr.<llm>.npy` to
    float32 precision.
    """
    sims = img_norm @ clip_emb_sel.T          # (N, K)
    proj = (sims @ llm_emb_sel).astype(np.float32, copy=False)  # (N, D_llm)
    return normalize_rows(proj)


def load_audit(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_trusted_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def require_existing(paths: list[Path], *, env_hint: str | None = None) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        msg = "Missing required phenotype files:\n  " + "\n  ".join(missing)
        if env_hint:
            msg += f"\nSet {env_hint} to the directory containing these files."
        raise SystemExit(msg)


def load_phecode_name_map(external_dir: Path) -> dict[str, str]:
    if PHENOTYPE_NAMES_CSV.exists():
        phen = pd.read_csv(PHENOTYPE_NAMES_CSV, dtype=str)
        phen["phecode"] = phen["phecode"].str.strip()
        phen["phecode_str"] = phen["phecode_str"].str.strip()
        return dict(zip(phen["phecode"], phen["phecode_str"]))

    fallback = external_dir / "phenotype__test__filtered_phecodes.csv"
    if fallback.exists():
        phen = pd.read_csv(fallback, dtype=str)
        phen["phecode"] = phen["phecode"].str.strip()
        phen["phecode_str"] = phen["phecode_str"].str.strip()
        return dict(zip(phen["phecode"], phen["phecode_str"]))

    raise SystemExit(
        "Missing phenotype name map. Expected either "
        f"{PHENOTYPE_NAMES_CSV} or {fallback}."
    )


def unique_keep_order(rows: list[dict], max_count: int) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        idx = int(row["concept_index"])
        if idx in seen:
            continue
        seen.add(idx)
        out.append(row)
        if max_count > 0 and len(out) >= max_count:
            break
    return out


def original_topk_set(label: str, audit: dict, bank: IndexedConceptBanks, top_k: int) -> ConceptSet:
    rows = audit.get(label, {}).get("positive", [])[:top_k]
    indices: list[int] = []
    concepts: list[str] = []
    scores: list[float] = []
    for row in rows:
        concept = str(row["concept"])
        idx = bank.concept_to_idx.get(concept)
        if idx is None:
            continue
        indices.append(idx)
        concepts.append(concept)
        scores.append(float(row.get("alignment", row.get("mean", 0.0))))
    return ConceptSet("original_topk", indices, concepts, ["original"] * len(indices), scores)


def trusted_set(
    label: str,
    summary_by_label: dict[str, dict],
    trusted_dir: Path,
    *,
    max_count: int,
    direct_only: bool,
) -> ConceptSet:
    row = summary_by_label.get(label, {})
    raw_csv = row.get("trusted_csv")
    fallback_path = trusted_dir / f"{slug(label)}.trusted_concepts.csv"
    if raw_csv:
        absolute_path = Path(raw_csv)
        basename_path = trusted_dir / absolute_path.name
        if absolute_path.exists():
            csv_path = absolute_path
        elif basename_path.exists():
            csv_path = basename_path
        else:
            csv_path = fallback_path
    else:
        csv_path = fallback_path
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


def coefficient_rows(
    *,
    label: str,
    phecode: str,
    method: str,
    model,
    cset: ConceptSet,
    top_k: int,
) -> list[dict]:
    if hasattr(model, "linear"):  # torch BinaryLinear
        coef = model.linear.weight.detach().cpu().numpy().reshape(-1)
    elif hasattr(model, "coef_"):  # sklearn LogisticRegression
        coef = np.asarray(model.coef_).reshape(-1)
    else:
        raise TypeError(f"Unsupported model type for coefficient extraction: {type(model)}")
    order = np.argsort(-np.abs(coef))[: min(top_k, len(coef))]
    rows = []
    for rank, j in enumerate(order, start=1):
        rows.append(
            {
                "label": label,
                "phecode": phecode,
                "method": method,
                "rank": rank,
                "concept_index": cset.indices[j],
                "concept": cset.concepts[j],
                "tier": cset.tiers[j],
                "concept_score": cset.scores[j],
                "coef": float(coef[j]),
                "abs_coef": float(abs(coef[j])),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    default_trusted_dir = ROOT / "outputs" / RUN_TAG / "audit" / "trusted_concept_spaces"
    default_audit_json = ROOT / "outputs" / RUN_TAG / "audit" / "phenotype__linear_openai__concept_importance.json"
    default_out_dir = EXTERNAL_DIR / "trusted_concept_probe"
    default_clip_bank = ROOT / f"concept_bank.clip_text_emb.{RUN_TAG}.npz"
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=RUN_TAG)
    ap.add_argument("--openai_concept_bank", default=str(ROOT / "concept_bank.openai_emb.npz"))
    ap.add_argument("--f2llm_concept_bank", default=str(ROOT / "concept_bank.f2llm_emb.npz"))
    ap.add_argument("--clip_concept_bank", default=str(default_clip_bank))
    ap.add_argument("--trusted_dir", default=str(default_trusted_dir))
    ap.add_argument("--audit_json", default=str(default_audit_json))
    ap.add_argument("--out_dir", default=str(default_out_dir))
    ap.add_argument("--labels", nargs="*", default=None, help="Exact phenotype names to run; default is all filtered labels.")
    ap.add_argument("--methods", nargs="*", default=["original_topk", "trusted", "direct"], choices=["original_topk", "trusted", "direct"])
    ap.add_argument("--feature_spaces", nargs="*", default=["clip", "openai"], choices=["clip", "openai", "f2llm"])
    ap.add_argument("--min_test_positives", type=int, default=50)
    ap.add_argument("--original_top_k", type=int, default=100, help="Number of current unrefined positive concepts to use.")
    ap.add_argument("--max_concepts", type=int, default=512, help="Cap trusted/direct concepts per label; 0 keeps all.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-8)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, etc.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_coef_k", type=int, default=20)
    ap.add_argument("--solver", choices=["adam", "lbfgs"], default="adam",
                    help="Linear-probe solver. lbfgs uses sklearn LogisticRegression on z-scored features.")
    ap.add_argument("--lbfgs_C", type=float, default=1e6,
                    help="L2 inverse strength for sklearn LBFGS. Default 1e6 = effectively unregularized.")
    ap.add_argument("--lbfgs_balanced", action="store_true",
                    help="Use class_weight='balanced' in sklearn LBFGS.")
    ap.add_argument("--lbfgs_max_iter", type=int, default=5000)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trusted_dir = Path(args.trusted_dir)
    audit = load_audit(Path(args.audit_json))
    summary_df = pd.read_csv(trusted_dir / "summary.csv")
    summary_by_label = {str(r["label"]): dict(r) for _, r in summary_df.iterrows()}

    print("=== Loading INSPECT phenotype splits ===")
    require_existing(
        [PHENOTYPE_LABELS, PHENOTYPE_MANIFEST],
        env_hint="INSPECT_PHENOTYPE_ROOT",
    )
    labels_df = pd.read_parquet(PHENOTYPE_LABELS)
    manifest = pd.read_parquet(PHENOTYPE_MANIFEST)
    phecode_to_str = load_phecode_name_map(EXTERNAL_DIR)

    print("=== Building cached 3D-CLIP image-feature lookup ===")
    lookup = build_inspect_lookup()

    test_manifest = manifest[(manifest["split"] == "test") & (manifest["visit_occurrence_id"].notna())]
    test_vols = [v for v in test_manifest["VolumeName"].values if v in lookup["vol_to_row"]]
    prev = labels_df.loc[test_vols].sum(axis=0)
    keep_phecodes = sorted(prev[prev >= args.min_test_positives].index.tolist())
    keep_names = [phecode_to_str.get(p, p) for p in keep_phecodes]
    label_to_phecode = dict(zip(keep_names, keep_phecodes))
    if args.labels:
        requested = set(args.labels)
        missing = sorted(requested - set(keep_names))
        if missing:
            raise SystemExit(f"Requested labels are not in the filtered phenotype set: {missing}")
        keep = [(p, n) for p, n in zip(keep_phecodes, keep_names) if n in requested]
        keep_phecodes = [p for p, _ in keep]
        keep_names = [n for _, n in keep]

    gathered = {}
    for split_name, split_label in [("train", "train"), ("valid", "validation"), ("test", "test")]:
        sub = gather_for_split(manifest[manifest["split"] == split_label], labels_df, keep_phecodes, lookup)
        if sub is None:
            raise SystemExit(f"No usable rows for split {split_name}")
        sub["img"] = normalize_rows(sub["img"])
        gathered[split_name] = sub
        print(f"  {split_name}: {len(sub['y'])} volumes")

    print("=== Loading indexed concept banks ===")
    # clip is always needed (concept-similarity scores); load each requested LLM space.
    llm_bank_arg = {"openai": args.openai_concept_bank, "f2llm": args.f2llm_concept_bank}
    bank_paths = {"clip": Path(args.clip_concept_bank)}
    for fs in args.feature_spaces:
        if fs == "clip":
            continue
        if fs not in llm_bank_arg:
            raise SystemExit(f"No concept-bank path wired for feature space '{fs}'")
        bank_paths[fs] = Path(llm_bank_arg[fs])
    bank = IndexedConceptBanks(bank_paths)

    result_rows: list[dict] = []
    top_coef_rows: list[dict] = []
    method_tags = [f"{feature_space}_{method}" for feature_space in args.feature_spaces for method in args.methods]
    prob_arrays = {
        method_tag: {
            split: np.full((len(gathered[split]["y"]), len(keep_names)), np.nan, dtype=np.float32)
            for split in gathered
        }
        for method_tag in method_tags
    }
    img_by_split = {split: gathered[split]["img"] for split in gathered}

    for label_i, label in enumerate(keep_names):
        phecode = label_to_phecode[label]
        y_train = gathered["train"]["y"][:, label_i].astype(np.float32)
        y_valid = gathered["valid"]["y"][:, label_i].astype(np.float32)
        y_test = gathered["test"]["y"][:, label_i].astype(np.float32)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"[skip] {label}: train/test has one class")
            continue

        candidate_sets: list[ConceptSet] = []
        if "original_topk" in args.methods:
            candidate_sets.append(original_topk_set(label, audit, bank, args.original_top_k))
        if "trusted" in args.methods:
            candidate_sets.append(trusted_set(label, summary_by_label, trusted_dir, max_count=args.max_concepts, direct_only=False))
        if "direct" in args.methods:
            candidate_sets.append(trusted_set(label, summary_by_label, trusted_dir, max_count=args.max_concepts, direct_only=True))

        for cset in candidate_sets:
            if not cset.indices:
                result_rows.append(
                    {
                        "label": label,
                        "phecode": phecode,
                        "basis": cset.method,
                        "n_concepts": 0,
                        "status": "skipped_no_concepts",
                    }
                )
                continue
            # clip scores are the shared first step of every LLM projection, so
            # select clip embeddings regardless of whether 'clip' is a probe space.
            clip_emb_sel = bank.select_embeddings(cset.indices, "clip")      # (K, 768)
            llm_emb_sel = {
                fs: bank.select_embeddings(cset.indices, fs)                 # (K, D_llm)
                for fs in args.feature_spaces if fs != "clip"
            }
            features_by_split = {fs: {} for fs in args.feature_spaces}
            for split in img_by_split:
                img_n = img_by_split[split]  # already L2-normalized
                if "clip" in args.feature_spaces:
                    features_by_split["clip"][split] = clip_concept_scores(img_n, clip_emb_sel)
                for fs, emb_sel in llm_emb_sel.items():
                    features_by_split[fs][split] = llm_concept_projection(img_n, clip_emb_sel, emb_sel)
            for feature_space in args.feature_spaces:
                method_tag = f"{feature_space}_{cset.method}"
                print(
                    f"[{label_i + 1:03d}/{len(keep_names):03d}] "
                    f"{label} / {method_tag}: {len(cset.indices)} concepts",
                    flush=True,
                )
                t0 = time.time()
                x_train = features_by_split[feature_space]["train"]
                x_valid = features_by_split[feature_space]["valid"]
                x_test = features_by_split[feature_space]["test"]
                if args.solver == "lbfgs":
                    x_train, x_valid, x_test = standardize_features(x_train, x_valid, x_test)
                    model, best_val, n_epochs = train_lbfgs_probe(
                        x_train, y_train, x_valid, y_valid,
                        C=args.lbfgs_C,
                        balanced=args.lbfgs_balanced,
                        max_iter=args.lbfgs_max_iter,
                    )
                    split_probs = {
                        "train": predict_lbfgs(model, x_train),
                        "valid": predict_lbfgs(model, x_valid),
                        "test": predict_lbfgs(model, x_test),
                    }
                else:
                    model, best_val, n_epochs = train_binary_probe(
                        x_train, y_train, x_valid, y_valid,
                        lr=args.lr,
                        weight_decay=args.weight_decay,
                        epochs=args.epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        device=device,
                    )
                    split_probs = {
                        "train": predict_binary(model, x_train, batch_size=args.batch_size, device=device),
                        "valid": predict_binary(model, x_valid, batch_size=args.batch_size, device=device),
                        "test": predict_binary(model, x_test, batch_size=args.batch_size, device=device),
                    }
                for split, probs in split_probs.items():
                    prob_arrays[method_tag][split][:, label_i] = probs
                row = {
                    "label": label,
                    "phecode": phecode,
                    "feature_space": feature_space,
                    "basis": cset.method,
                    "method": method_tag,
                    "n_concepts": len(cset.indices),
                    "n_direct": int(sum(t == "direct" for t in cset.tiers)),
                    "n_associated": int(sum(t == "associated" for t in cset.tiers)),
                    "n_original": int(sum(t == "original" for t in cset.tiers)),
                    "profile_status": summary_by_label.get(label, {}).get("profile_status", ""),
                    "radiology_flag": summary_by_label.get(label, {}).get("radiology_flag", ""),
                    "train_auc": safe_auc(y_train, split_probs["train"]),
                    "valid_auc": safe_auc(y_valid, split_probs["valid"]),
                    "test_auc": safe_auc(y_test, split_probs["test"]),
                    "test_ap": safe_ap(y_test, split_probs["test"]),
                    "test_brier": safe_brier(y_test, split_probs["test"]),
                    "best_valid_score": best_val,
                    "epochs": n_epochs,
                    "seconds": round(time.time() - t0, 2),
                    "status": "ok",
                }
                result_rows.append(row)
                # Per-concept coefficients only make sense for the clip path,
                # where each probe weight maps 1:1 to a selected concept.
                # The openai path probes a 3072-d OpenAI-space embedding, so
                # weights are per-OpenAI-dim, not per-concept.
                if feature_space == "clip":
                    top_coef_rows.extend(
                        coefficient_rows(
                            label=label,
                            phecode=phecode,
                            method=method_tag,
                            model=model,
                            cset=cset,
                            top_k=args.top_coef_k,
                        )
                    )
                print(f"    test_auc={row['test_auc']:.4f} test_ap={row['test_ap']:.4f} ({row['seconds']}s)", flush=True)

    summary = pd.DataFrame(result_rows)
    top_coef = pd.DataFrame(top_coef_rows)
    summary_path = out_dir / "trusted_concept_probe_summary.csv"
    top_coef_path = out_dir / "trusted_concept_probe_top_coefficients.csv"
    overall_path = out_dir / "trusted_concept_probe_overall.csv"
    summary.to_csv(summary_path, index=False)
    top_coef.to_csv(top_coef_path, index=False)

    ok = summary[summary["status"] == "ok"].copy() if not summary.empty else summary
    overall_rows = []
    if not ok.empty:
        for method, df_m in ok.groupby("method"):
            overall_rows.append(
                {
                    "method": method,
                    "n_labels": int(len(df_m)),
                    "mean_test_auc": finite_mean(df_m["test_auc"]),
                    "mean_test_ap": finite_mean(df_m["test_ap"]),
                    "mean_test_brier": finite_mean(df_m["test_brier"]),
                    "median_n_concepts": float(df_m["n_concepts"].median()),
                }
            )
    pd.DataFrame(overall_rows).to_csv(overall_path, index=False)

    for method, splits in prob_arrays.items():
        for split, arr in splits.items():
            np.savez_compressed(
                out_dir / f"{split}__{method}__probs.npz",
                probs=arr,
                labels=np.asarray(keep_names, dtype=object),
                phecodes=np.asarray(keep_phecodes, dtype=object),
                volumes=np.asarray(gathered[split]["vols"], dtype=object),
            )

    manifest_payload = {
        "tag": args.tag,
        "concept_banks": {fs: str(p) for fs, p in bank_paths.items()},
        "openai_concept_bank": str(Path(args.openai_concept_bank)),
        "clip_concept_bank": str(Path(args.clip_concept_bank)),
        "trusted_dir": str(trusted_dir),
        "audit_json": str(Path(args.audit_json)),
        "methods": args.methods,
        "feature_spaces": args.feature_spaces,
        "labels": keep_names,
        "phecodes": keep_phecodes,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "device": str(device),
        "original_top_k": args.original_top_k,
        "max_concepts": args.max_concepts,
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest_payload, f, indent=2)

    print(f"\nWrote {summary_path}")
    print(f"Wrote {overall_path}")
    print(f"Wrote {top_coef_path}")
    if overall_rows:
        print(pd.DataFrame(overall_rows).to_string(index=False))


if __name__ == "__main__":
    main()

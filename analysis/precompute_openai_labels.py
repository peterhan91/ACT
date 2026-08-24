#!/usr/bin/env python3
"""
Precompute OpenAI text-embedding-3-large label embeddings.

Why this exists: the GH200 compute nodes block outbound HTTPS to
`api.openai.com`, so the eval scripts cannot embed pos/neg label prompts at
run time. This script computes every label embedding the pipeline needs and
writes them to the per-RUN_TAG cache directories so the eval scripts pick
them up without any API access.

Run this on ANY machine with internet + `OPENAI_API_KEY`, then commit/push
the resulting `outputs/<tag>/cache/label_emb.openai*.npz` files (each a few
KB), or scp them onto the cluster.

Files produced per --tags entry:
  outputs/<tag>/cache/label_emb.openai.npz             — 18 CT-RATE labels
  outputs/<tag>/cache/label_emb.openai.<sig>.npz       — sig-keyed lists for
      the 3 INSPECT PE tasks (snake_case + verbose) and the 221 filtered
      phecode names (taken from any tag's phenotype__test__filtered_phecodes.csv;
      the list is identical across checkpoints because it depends only on
      the manifest test set, not the image encoder).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from clear3d.paths import (
    LABELS_18, LABELS_INSPECT_PROGNOSIS_PROMPTS, LABELS_PE,
    LABELS_RSNA2023_TRAUMA_PROMPTS, NSCLC_LABELSET_PROMPTS,
)
from clear3d.llm_prompts import _encode_with_openai

PE_PRETTY = [
    "pulmonary embolism",
    "acute pulmonary embolism",
    "subsegmental pulmonary embolism",
]

OPENAI_MODEL = "text-embedding-3-large"


def _sig(labels: list[str]) -> str:
    return hashlib.sha1("|".join(labels).encode("utf-8")).hexdigest()[:10]


def encode_and_save(out_path: Path, labels: list[str], *, overwrite: bool = False):
    """pos = label.lower(); neg = 'no ' + label.lower(); concat-encode, then split."""
    if out_path.exists() and not overwrite:
        print(f"  cached: {out_path.name}  ({2*len(labels)} prompts)")
        return
    pos_texts = [lbl.lower() for lbl in labels]
    neg_texts = [f"no {lbl.lower()}" for lbl in labels]
    print(f"  encoding {2*len(labels)} prompts → {out_path.name} ...")
    all_emb = _encode_with_openai(OPENAI_MODEL, pos_texts + neg_texts)
    pos = all_emb[: len(labels)]
    neg = all_emb[len(labels):]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, pos=pos, neg=neg, labels=np.asarray(labels, dtype=object))


def find_phecode_names(repo_root: Path) -> list[str] | None:
    """Look for any phenotype__test__filtered_phecodes.csv across tags. The
    filtered list depends only on the manifest test set (>=50 positives) and
    is the same across image encoders, so any saved CSV works."""
    candidates = list(repo_root.glob("outputs/*/external/phenotype__test__filtered_phecodes.csv"))
    if not candidates:
        return None
    df = pd.read_csv(candidates[0])
    print(f"  using phecode list from {candidates[0]} ({len(df)} phecodes)")
    return df["phecode_str"].astype(str).str.strip().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tags", nargs="*",
        default=["dinov3", "v1", "transformer", "attentive"],
        help="run tags to populate (creates outputs/<tag>/cache/ if needed)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set in env; aborting.")

    phecode_names = find_phecode_names(ROOT)
    if phecode_names is None:
        print("(no phenotype__test__filtered_phecodes.csv found; skipping 221-label phenotype embedding — "
              "run exp_phenotype_ct.py once first to produce it, then rerun this script)")

    for tag in args.tags:
        cache = ROOT / "outputs" / tag / "cache"
        print(f"\n=== tag: {tag}  →  {cache} ===")
        encode_and_save(cache / "label_emb.openai.npz", LABELS_18,
                        overwrite=args.overwrite)
        nsclc_specs = [
            (prompts, f"NSCLC {labelset}")
            for labelset, prompts in NSCLC_LABELSET_PROMPTS.items()
        ]
        for spec_labels, descr in [
            (LABELS_PE, "INSPECT PE (snake_case)"),
            (PE_PRETTY, "INSPECT PE (verbose)"),
            (LABELS_RSNA2023_TRAUMA_PROMPTS, "RSNA-2023 trauma (verbose)"),
            (LABELS_INSPECT_PROGNOSIS_PROMPTS, "INSPECT prognosis (7 binaries)"),
            *nsclc_specs,
        ]:
            sig = _sig(spec_labels)
            encode_and_save(cache / f"label_emb.openai.{sig}.npz",
                            spec_labels, overwrite=args.overwrite)
        if phecode_names:
            sig = _sig(phecode_names)
            encode_and_save(cache / f"label_emb.openai.{sig}.npz",
                            phecode_names, overwrite=args.overwrite)

    print("\nDone. Commit/push the new files in outputs/<tag>/cache/, then re-run "
          "the GH200 pipeline with openai included.")


if __name__ == "__main__":
    main()

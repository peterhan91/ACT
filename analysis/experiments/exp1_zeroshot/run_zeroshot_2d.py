#!/usr/bin/env python3
"""
exp1_zeroshot — shared runner for the 2D baselines (MedSigLIP / BiomedCLIP /
OpenAI CLIP) on zero-shot finding classification. 2D slices through each model's
OWN image pipeline, average-pooled; label prompts via the model's text tower;
identical `softmax_pos_neg` scoring. 2D models have no CT-native preprocessing,
so their own image pipeline IS their native input -> reported in the native
column (mode="native"), like `ours`. `--model NAME`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # exp1_zeroshot
ROOT = HERE.parent                               # CLEAR_CT_EXPS
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CLEAR3D_REPO", "/path/to/ACT/analysis")

import common as C                 # noqa: E402
import models2d as M2              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(M2.CONFIG))
    ap.add_argument("--datasets", nargs="*",
                    default=["ctrate_test", "radchest", "rsna2023_test"], choices=list(C.SPECS))
    ap.add_argument("--source", choices=["native", "h5"], default="native",
                    help="native = original full-res scans (official pipeline, headline); "
                         "h5 = our preprocessed 224^3 voxels (controlled ablation)")
    ap.add_argument("--limit", type=int, default=0, help="cap volumes/dataset (smoke/timing)")
    # Parallel sharding for inspect_pheno (the 2D encode is single-threaded; shard it
    # across processes to fill the GPU). --shard i with --nshards N encodes the i-th
    # contiguous block to its own cache; --finalize concatenates all N shards + scores.
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=-1, help=">=0 selects a contiguous block")
    ap.add_argument("--finalize", action="store_true", help="combine inspect_pheno shards + score")
    args = ap.parse_args()
    print(f"Model(2D): {args.model} ({M2.CONFIG[args.model]})  datasets={args.datasets}")
    enc = M2.load(args.model)
    RESULTS = HERE / args.model / "results"
    summary = []
    for ds in args.datasets:
        try:
            if ds == "inspect_pheno":
                import math
                import numpy as np
                man, _, y, labels, prompts = C.load_dataset(ds)
                N = max(1, args.nshards)
                if args.shard >= 0 and not args.finalize:
                    # SHARD: encode this contiguous block to its own cache, no scoring.
                    per = math.ceil(len(man) / N)
                    lo, hi = args.shard * per, min((args.shard + 1) * per, len(man))
                    M2.inspect_native_feats(
                        enc, args.model, man.iloc[lo:hi],
                        cache_name=f"{args.model}__inspect_pheno__shard{args.shard}of{N}.npy")
                    print(f"[shard {args.shard}/{N}] vols {lo}:{hi} ({hi-lo}) encoded", flush=True)
                    continue
                if args.finalize:
                    img = np.concatenate([np.load(
                        M2.CACHE / f"{args.model}__inspect_pheno__shard{i}of{N}.npy")
                        for i in range(N)], axis=0)
                    if len(img) != len(y):
                        raise ValueError(f"finalize: img {len(img)} != y {len(y)}")
                else:
                    if args.limit:
                        man, y = man.head(args.limit), y[:args.limit]
                    img = M2.inspect_native_feats(enc, args.model, man)
                t0 = time.time()
            else:
                _, h5_idx, y, labels, prompts = C.load_dataset(ds)
                if args.limit:
                    h5_idx, y = h5_idx[:args.limit], y[:args.limit]
                t0 = time.time()
                if args.source == "native":
                    img = M2.native_volume_feats(enc, args.model, ds, limit=args.limit)
                else:
                    img = M2.dataset_image_feats(enc, args.model, ds, C.h5_path(ds), h5_idx)
            # Each 2D model's OWN official zero-shot caption template (CLIP
            # "a photo of a {}", BiomedCLIP "this is a photo of {}", MedSigLIP
            # "a photo of {}") — bare class words are OOD for these text towers.
            # Kept as a pos/neg pair so scoring stays identical to ours/3D baselines.
            tmpl = getattr(enc, "prompt_template", "{}")
            pos = enc.encode_texts([tmpl.format(p) for p in prompts])
            neg = enc.encode_texts([tmpl.format(f"no {p}") for p in prompts])
            probs = C.softmax_pos_neg(img, pos, neg)
            row = C.save_run(args.model, M2.CONFIG[args.model], ds, args.source, probs, y, labels,
                             len(y), RESULTS, time.time() - t0,
                             extra={"input_source": f"2d_own_pipeline_{args.source}",
                                    "ct_windows": M2.WINDOW_NAME, "n_slices": M2.N_SLICES,
                                    "prompt_template": tmpl})
            summary.append(row)
        except Exception as e:  # one dataset failing must not kill the rest
            print(f"[skip {args.model}/{ds}] {type(e).__name__}: {e}", flush=True)
    C.write_summary(summary, RESULTS, f"{args.model} (2D+avgpool) — exp1 zero-shot (native)")


if __name__ == "__main__":
    main()

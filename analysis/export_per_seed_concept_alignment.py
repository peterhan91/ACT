# LEGACY ALIGNMENT CONVENTION. This exporter reuses the normalized-cosine
# machinery of concept_audit_20seeds.py, so the per-seed values it dumps are
# cosines of NORMALIZED probe weights, NOT the manuscript's Equation 2
# alignment score. No reported alignment number comes from this file. The
# authoritative raw-weight exports live in
# analysis/experiments/exp4_confounder_audit/.
"""Export the raw per-seed (20-run) concept-alignment datapoints.

concept_audit_20seeds.py computes, for each (label, concept), the cosine alignment
of that seed's probe weight with the concept embedding — but only persists the
mean/std (in __20seeds_meta.json). This dumps the underlying 20 per-seed values so
you can draw the per-seed scatter behind each concept's mean±s.d.

Faithful to the audit: same _load_seed_W, concept centering, _align, and top-K
selection (top-K pos + top-K neg per label by the alignment of the mean-W). The
full (20 × n_labels × 376k-concept) tensor is ~6.6 GB, so we only materialize the
audit's top-K concepts per label (pass --labels to narrow further for a figure).

Usage:
  CLIP3D_RUN_TAG=v1 python export_per_seed_concept_alignment.py \
      --task phenotype --llm f2llm --bestlr lbfgs \
      [--labels Shock "Respiratory insufficiency"] [--top_k 100] \
      [--out outputs/v1/audit/phenotype__linear_f2llm__per_seed_alignment.csv]

Output long CSV columns:
  label, sign, rank, concept, seed, alignment,   # one row per (label, concept, seed)
  seed_mean, seed_std, align_of_meanW            # repeated per group for convenience
"""
import argparse
import numpy as np
import pandas as pd

from concept_audit_20seeds import TASKS, AUDIT_DIR, _load_seed_W, _streaming_sims
from clear3d.projection import load_concept_bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="phenotype", choices=list(TASKS))
    ap.add_argument("--llm", default="f2llm",
                    choices=["openai", "gteqwen2", "sfr", "harrier", "f2llm"])
    ap.add_argument("--bestlr", default="lbfgs",
                    help="lr tag of the per-seed coefs to load (lbfgs = sklearn L-BFGS)")
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--top_k", type=int, default=100,
                    help="top-K positive + top-K negative concepts per label to dump")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="restrict to these label names (default: all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = TASKS[args.task]
    dir_ = spec["dir"]
    prefix = spec["prefix"].format(llm=args.llm)
    center = spec["default_center"]

    Ws, labels, meta = _load_seed_W(dir_, prefix, args.bestlr, n_seeds=args.n_seeds)
    print(f"loaded {meta['n_seeds']} seeds | W = {Ws.shape} | lr={args.bestlr} center={center}")

    # Optionally subset labels (cheaper + smaller output) — do it before the sims pass.
    if args.labels:
        idx = [labels.index(l) for l in args.labels]  # raises if a name is wrong
        Ws = Ws[:, idx, :]
        labels = list(args.labels)
        print(f"restricted to {len(labels)} label(s): {labels}")

    bank = load_concept_bank(llm=args.llm, device="cpu")
    concepts = bank["concepts"]
    e = bank["llm_emb"].numpy()  # L2-normalized
    if center:
        e = e - e.mean(axis=0, keepdims=True)
        e = e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-9)

    # Reuse the audit's streaming pass: `needed` is {(li, ci): np.ndarray[S]} — the
    # raw per-seed alignment values for the top-K pos/neg concepts of the mean-W.
    sims_mean, needed, _ = _streaming_sims(
        Ws, e, labels, concepts,
        k_payload=args.top_k, k_stability=10, weight_field=spec["weight_field"],
    )

    # Rank the selected concepts within each label/sign (by mean-W alignment), as the audit does.
    rank_of = {}
    for li in range(len(labels)):
        v = sims_mean[li]
        pos = np.where(v > 0)[0]; neg = np.where(v < 0)[0]
        for r, ci in enumerate(pos[np.argsort(-v[pos])][:args.top_k], 1):
            rank_of[(li, int(ci))] = ("positive", r)
        for r, ci in enumerate(neg[np.argsort(v[neg])][:args.top_k], 1):
            rank_of[(li, int(ci))] = ("negative", r)

    rows = []
    for (li, ci), seed_vals in needed.items():
        sign, rank = rank_of.get((li, ci), ("", 0))
        smean, sstd = float(seed_vals.mean()), float(seed_vals.std(ddof=1))
        align_mw = float(sims_mean[li, ci])
        for s, val in enumerate(seed_vals, start=1):
            rows.append({
                "label": labels[li], "sign": sign, "rank": rank,
                "concept": concepts[ci], "seed": s, "alignment": float(val),
                "seed_mean": smean, "seed_std": sstd, "align_of_meanW": align_mw,
            })

    df = pd.DataFrame(rows).sort_values(["label", "sign", "rank", "seed"]).reset_index(drop=True)
    out = args.out or str(AUDIT_DIR / f"{spec['out_basename'].format(llm=args.llm)}__per_seed_alignment.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows = {df[['label','concept']].drop_duplicates().shape[0]} concept-groups x {args.n_seeds} seeds)")


if __name__ == "__main__":
    main()

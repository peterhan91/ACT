#!/usr/bin/env python3
"""Concept-latent geometry of the LLM concept bank (GenePT-style).

Pipeline (Chen & Zou, GenePT, Nat Biomed Eng 2024 recipe):
    L2-normalize -> PCA(n_components=50) -> UMAP(min_dist=0.5, spread=1)
run on the FULL concept bank (no subsampling). Concepts are tagged with a
clinical-category regex taxonomy (priority-ordered, first match wins; rare/
specific groups like Abdominal trauma first so they are not absorbed). We then
run MiniBatchKMeans on the PCA-50 space and report adjusted Rand index (aRI)
and adjusted mutual info (aMI) vs the category labels -- the GenePT quantitative
check that the latent captures meaningful structure.

Default embedder: openai = text-embedding-3-large (3072-d).

Outputs (outputs/concept_umap/<llm>/):
    umap2d.npy            (N, 2) float32   2-D coordinates, full bank
    categories.npy        (N,)  object     clinical category per concept
    pca50.npy             (N,50) float32   (optional, --save_pca)
    metrics.json          counts per category + kMeans aRI/aMI
    concept_latent.png    the figure
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------- taxonomy
# (label, regex)  -- evaluated in order; FIRST match wins. Patterns are matched
# against the lowercased concept text. Specific/rare groups come first.
TAXONOMY = [
    ("Abdominal trauma",        r"laceration|hematoma|active extravasation|hemoperitoneum|splenic injur|hepatic injur|renal injur|bowel injur|mesenteric injur|organ injur|traumatic|contusion"),
    ("Pulmonary embolism",      r"pulmonary embol|embolus|filling defect|thromboemboli|thrombus in the"),
    ("Pneumothorax",            r"pneumothorax"),
    ("Pleural effusion",        r"pleural effusion|pleural fluid"),
    ("Ascites/free fluid",      r"ascites|peritoneal fluid|free fluid|perihepatic fluid|free intraperitoneal|abdominal effusion|intra-abdominal effusion"),
    ("Atherosclerosis",         r"atheroma|atheroscler|calcified plaque|vascular calcific|coronary calcif|arterial calcific"),
    ("Aortic aneurysm/dissect", r"aneurysm|aortic dilat|\bectasia|dissection"),
    ("Bronchiectasis/airway",   r"bronchiectasis|peribronchial|bronchial wall thick|airway"),
    ("Emphysema",               r"emphysema"),
    ("Interstitial/fibrosis",   r"fibrosis|interstitial|septal thickening|reticulation|honeycomb"),
    ("Lymphadenopathy",         r"lymphadenopathy|lymph node|enlarged node|mediastinal node"),
    ("Cardiomegaly/pericardial",r"cardiomegaly|pericardial effusion|enlarged heart|cardiac enlarg"),
    ("Hepatic",                 r"hepatic|\bliver|cirrhosis|steatosis|fatty liver"),
    ("Renal",                   r"\brenal|kidney|nephrolith|hydronephrosis"),
    ("Biliary/GB",              r"biliary|bile duct|gallstone|cholelith|gallbladder|cholecystitis"),
    ("Fracture/bone",           r"fracture|osseous|vertebral compression|lytic lesion|sclerotic lesion|\bbony"),
    ("Airspace (PNA/GGO)",      r"consolidation|ground.?glass|airspace|pneumoni|infiltrat"),
    ("Nodule/mass",             r"nodule|\bmass|spiculat|metasta|\blesion"),
]
OTHER = "other"


def categorize(concepts: np.ndarray) -> np.ndarray:
    compiled = [(name, re.compile(pat)) for name, pat in TAXONOMY]
    out = np.empty(len(concepts), dtype=object)
    for i, c in enumerate(concepts):
        cl = str(c).lower()
        lab = OTHER
        for name, rx in compiled:
            if rx.search(cl):
                lab = name
                break
        out[i] = lab
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="openai")
    ap.add_argument("--bank", default=None, help="override path to concept_bank.<llm>_emb.npz")
    ap.add_argument("--n_pca", type=int, default=50)
    ap.add_argument("--min_dist", type=float, default=0.5)
    ap.add_argument("--spread", type=float, default=1.0)
    ap.add_argument("--n_neighbors", type=int, default=15)
    ap.add_argument("--seed", type=int, default=-1,
                    help=">=0 sets UMAP random_state (reproducible, single-thread); "
                         "-1 lets UMAP parallelize across cores (faster, non-deterministic)")
    ap.add_argument("--save_pca", action="store_true")
    ap.add_argument("--out", default="outputs/concept_umap")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    root = Path("/path/to/ACT/analysis")
    bank = Path(args.bank) if args.bank else root / f"concept_bank.{args.llm}_emb.npz"
    outdir = root / args.out / args.llm
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"=== concept-latent UMAP | llm={args.llm} | {bank} ===", flush=True)
    z = np.load(bank, allow_pickle=True)
    emb = z["emb"].astype(np.float32)            # (N, D)
    concepts = z["concepts"]
    N, D = emb.shape
    print(f"  loaded {N:,} concepts x {D} dims  [{time.time()-t0:.0f}s]", flush=True)

    # L2-normalize (cosine geometry)
    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    np.divide(emb, np.clip(nrm, 1e-9, None), out=emb)

    # categories
    cats = categorize(concepts)
    uniq, counts = np.unique(cats, return_counts=True)
    cat_counts = {str(u): int(c) for u, c in sorted(zip(uniq, counts), key=lambda x: -x[1])}
    print(f"  categorized [{time.time()-t0:.0f}s]: "
          + ", ".join(f"{k}={v}" for k, v in list(cat_counts.items())[:6]) + " ...", flush=True)

    # PCA -> n_pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=args.n_pca, svd_solver="randomized", random_state=0)
    X = pca.fit_transform(emb).astype(np.float32)
    print(f"  PCA -> {X.shape} | var explained {pca.explained_variance_ratio_.sum():.3f}"
          f"  [{time.time()-t0:.0f}s]", flush=True)
    del emb

    # UMAP
    import umap
    kw = dict(n_components=2, n_neighbors=args.n_neighbors,
              min_dist=args.min_dist, spread=args.spread, metric="euclidean", verbose=True)
    if args.seed >= 0:
        kw["random_state"] = args.seed
    reducer = umap.UMAP(**kw)
    Y = reducer.fit_transform(X).astype(np.float32)
    print(f"  UMAP -> {Y.shape}  [{time.time()-t0:.0f}s]", flush=True)

    # kMeans aRI/aMI vs categories (labeled subset only)
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    lab_mask = cats != OTHER
    k = len(TAXONOMY)
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096, n_init=3)
    km_lab = km.fit_predict(X[lab_mask])
    ari = float(adjusted_rand_score(cats[lab_mask], km_lab))
    ami = float(adjusted_mutual_info_score(cats[lab_mask], km_lab))
    print(f"  kMeans(k={k}) on labeled subset (n={int(lab_mask.sum()):,}): "
          f"aRI={ari:.3f}  aMI={ami:.3f}", flush=True)

    # save
    np.save(outdir / "umap2d.npy", Y)
    np.save(outdir / "categories.npy", cats)
    if args.save_pca:
        np.save(outdir / "pca50.npy", X)
    metrics = dict(llm=args.llm, n_concepts=int(N), dim=int(D),
                   n_pca=args.n_pca, pca_var=float(pca.explained_variance_ratio_.sum()),
                   umap=dict(min_dist=args.min_dist, spread=args.spread,
                             n_neighbors=args.n_neighbors, seed=args.seed),
                   category_counts=cat_counts,
                   kmeans=dict(k=k, n_labeled=int(lab_mask.sum()), aRI=ari, aMI=ami),
                   seconds=round(time.time() - t0, 1))
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  wrote {outdir}/umap2d.npy, categories.npy, metrics.json", flush=True)

    if not args.no_plot:
        plot(Y, cats, outdir, args.llm, cat_counts)
    print(f"=== done in {time.time()-t0:.0f}s ===", flush=True)


def plot(Y, cats, outdir, llm, cat_counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # category color order = by descending count, but keep Abdominal trauma drawn last/on top
    cat_order = [c for c in cat_counts if c != OTHER]
    palette = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors
    color = {c: palette[i % len(palette)] for i, c in enumerate(cat_order)}
    color["Abdominal trauma"] = (0.86, 0.08, 0.12)  # crimson, emphasized

    fig, ax = plt.subplots(figsize=(11, 9))
    other = cats == OTHER
    ax.scatter(Y[other, 0], Y[other, 1], s=0.3, c="0.85", alpha=0.05,
               linewidths=0, rasterized=True)
    draw_last = "Abdominal trauma"
    for c in [x for x in cat_order if x != draw_last] + [draw_last]:
        m = cats == c
        if not m.any():
            continue
        emph = c == draw_last
        ax.scatter(Y[m, 0], Y[m, 1], s=4 if emph else 1.2,
                   c=[color[c]], alpha=0.9 if emph else 0.45,
                   linewidths=0, rasterized=True,
                   label=f"{c} ({cat_counts[c]:,})", zorder=5 if emph else 3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Concept latent ({llm}): {len(cats):,} concepts · "
                 f"PCA(50)→UMAP(min_dist=0.5)", fontsize=12)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                    fontsize=7, markerscale=3, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "concept_latent.png", dpi=200, bbox_inches="tight")
    print(f"  wrote {outdir}/concept_latent.png", flush=True)


if __name__ == "__main__":
    main()

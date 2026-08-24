#!/usr/bin/env python3
"""
Baseline comparison for the concept-space quality metric, per reviewer-safe design:
compare f2llm ONLY against weak / null encoders (random, the CLIP text tower, and
BiomedBERT) -- NOT against other strong LLM embedders (that would look cherry-picked).

Every bank goes through the identical pipeline: L2-normalize -> PCA-50 -> the same
MiniBatchKMeans(aRI/aMI) and kNN-purity used for the main figure, so the metric is
dimension-controlled (all scored in a common 50-D space).

Idempotent: computes metrics for whatever banks are present, merges with the cached
f2llm numbers and the random control, and rewrites the comparison store. Run once now
(CLIP-text variants) and again after BiomedBERT embedding finishes.
"""
import json, csv, time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from concept_space_extended import kmeans_ari_ami, knn_purity, UMAP_DIR, OUT, OTHER

ROOT = Path(__file__).resolve().parent
CATS = np.load(UMAP_DIR / "f2llm" / "categories.npy", allow_pickle=True)

BANKS = {
    "clip_text.v1":          "concept_bank.clip_text_emb.v1.npz",
    "clip_text.transformer": "concept_bank.clip_text_emb.transformer.npz",
    "clip_text.attentive":   "concept_bank.clip_text_emb.attentive.npz",
    "clip_text.dinov3":      "concept_bank.clip_text_emb.dinov3.npz",
    "biomedbert":            "concept_bank.biomedbert_emb.npz",
}


def pca50_metrics(emb):
    emb = emb.astype(np.float32, copy=True)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    pca = PCA(n_components=50, svd_solver="randomized", random_state=0)
    X = pca.fit_transform(emb).astype(np.float32)
    ari, ami = kmeans_ari_ami(X, CATS, k=18)
    _, pur = knn_purity(X, CATS, k=15)
    return dict(dim=int(emb.shape[1]), pca_var=round(float(pca.explained_variance_ratio_.sum()), 4),
                aRI=round(ari, 4), aMI=round(ami, 4), knn_purity=round(pur, 4))


def main():
    store_path = OUT / "baseline_comparison.json"
    store = json.loads(store_path.read_text()) if store_path.exists() else {}

    # anchor rows: cached f2llm + random control (already computed)
    met = json.loads((UMAP_DIR / "f2llm" / "metrics.json").read_text())
    pur = json.loads((UMAP_DIR / "f2llm" / "knn_purity.json").read_text())
    store["f2llm"] = dict(dim=met["dim"], pca_var=round(met["pca_var"], 4),
                          aRI=round(met["kmeans"]["aRI"], 4), aMI=round(met["kmeans"]["aMI"], 4),
                          knn_purity=round(pur["overall"], 4))
    rc = json.loads((OUT / "random_control.json").read_text())
    store["random"] = dict(dim=50, pca_var=None, **{k: rc["random_gaussian"][k] for k in ("aRI", "aMI", "knn_purity")})

    for name, fn in BANKS.items():
        p = ROOT / fn
        if not p.exists():
            print(f"  skip {name}: {fn} not present yet")
            continue
        t0 = time.time()
        emb = np.load(p, allow_pickle=True)["emb"]
        store[name] = pca50_metrics(emb)
        del emb
        print(f"  {name:24} {store[name]}  [{time.time()-t0:.0f}s]")

    store_path.write_text(json.dumps(store, indent=2))
    # tidy CSV, ordered by aMI
    rows = [dict(encoder=k, **v) for k, v in store.items()]
    rows.sort(key=lambda r: (r["aMI"] if r["aMI"] is not None else -1), reverse=True)
    with open(OUT / "baseline_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["encoder", "dim", "pca_var", "aRI", "aMI", "knn_purity"])
        w.writeheader(); w.writerows(rows)
    print("\n=== baseline comparison (aMI-sorted) ===")
    for r in rows:
        print(f"  {r['encoder']:24} dim={str(r['dim']):>5}  aRI={r['aRI']}  aMI={r['aMI']}  purity={r['knn_purity']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extended concept-space analysis for Figure 3 (CLEAR-3D).

Adds, on top of the existing UMAP + per-family kNN purity, three GenePT/TITAN-style
validations of the f2llm concept embedding space:

  (1) CROSS-EMBEDDER comparison  -- reads the cached metrics for all five text
      embedders (f2llm, openai, sfr, harrier, gteqwen2): aRI, aMI, overall kNN
      purity, PCA-50 variance. Pure read of outputs/concept_umap/<llm>/.

  (2) RANDOM-EMBEDDING negative control -- generates iid Gaussian vectors of the
      same PCA dimensionality, runs the SAME MiniBatchKMeans(aRI/aMI) and kNN
      purity pipeline against the real family labels. Shows the real space is far
      above chance (rules out a dimensionality artefact).

  (3) NEAREST-NEIGHBOUR clinical relationships -- for a set of anchor findings,
      reports the top cosine neighbours in the raw f2llm space (with similarities
      and family labels), demonstrating recovery of known clinical relationships.

Everything is READ-ONLY on existing artifacts and WRITES only to
outputs/concept_umap/extended/ . Does not require umap-learn or pca50.npy.
"""
import json, csv, time, re
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
UMAP_DIR = ROOT / "outputs" / "concept_umap"
OUT = UMAP_DIR / "extended"
OUT.mkdir(parents=True, exist_ok=True)
OTHER = "other"
EMBEDDERS = ["f2llm", "openai", "sfr", "harrier", "gteqwen2"]
SEED = 0


def knn_purity(X, cats, k=15, sample_cap=1500, seed=0):
    """Faithful copy of concept_latent_plots.knn_purity (macro-avg over families)."""
    lab = cats != OTHER
    Xl, cl = X[lab], cats[lab]
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(Xl)
    rng = np.random.RandomState(seed)
    out = {}
    for c in np.unique(cl):
        idx = np.where(cl == c)[0]
        if len(idx) > sample_cap:
            idx = rng.choice(idx, sample_cap, replace=False)
        _, nbr = nn.kneighbors(Xl[idx])
        nbr = nbr[:, 1:]
        same = (cl[nbr] == c).mean()
        out[str(c)] = dict(purity=float(same), n=int((cl == c).sum()))
    overall = float(np.mean([v["purity"] for v in out.values()]))
    return out, overall


def kmeans_ari_ami(X, cats, k=18):
    """Faithful copy of the labeled-subset kMeans block in concept_latent_umap.py."""
    lab = cats != OTHER
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096, n_init=3)
    lab_pred = km.fit_predict(X[lab])
    return (float(adjusted_rand_score(cats[lab], lab_pred)),
            float(adjusted_mutual_info_score(cats[lab], lab_pred)))


# ---------------------------------------------------------------- (1) cross-embedder
def cross_embedder():
    rows = []
    for m in EMBEDDERS:
        d = UMAP_DIR / m
        met = json.loads((d / "metrics.json").read_text())
        pur = json.loads((d / "knn_purity.json").read_text())
        rows.append(dict(
            embedder=m, dim=met.get("dim"), pca_var=round(met.get("pca_var", float("nan")), 4),
            aRI=round(met["kmeans"]["aRI"], 4), aMI=round(met["kmeans"]["aMI"], 4),
            knn_purity=round(pur.get("overall", float("nan")), 4),
            n_labeled=met["kmeans"]["n_labeled"]))
    rows.sort(key=lambda r: -r["aMI"])
    with open(OUT / "cross_embedder_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n=== (1) Cross-embedder comparison (sorted by aMI) ===")
    print(f"{'embedder':10} {'dim':>5} {'pca_var':>8} {'aRI':>7} {'aMI':>7} {'purity':>7}")
    for r in rows:
        print(f"{r['embedder']:10} {r['dim']:>5} {r['pca_var']:>8} {r['aRI']:>7} {r['aMI']:>7} {r['knn_purity']:>7}")
    return rows


# ---------------------------------------------------------------- (2) random control
def random_control():
    cats = np.load(UMAP_DIR / "f2llm" / "categories.npy", allow_pickle=True)
    n_lab = int((cats != OTHER).sum())
    # real (cached) f2llm numbers
    met = json.loads((UMAP_DIR / "f2llm" / "metrics.json").read_text())
    pur = json.loads((UMAP_DIR / "f2llm" / "knn_purity.json").read_text())
    real = dict(aRI=met["kmeans"]["aRI"], aMI=met["kmeans"]["aMI"], knn_purity=pur["overall"])
    # random: iid Gaussian at the PCA-50 dimensionality, L2-normalized (matches metric space)
    rng = np.random.RandomState(SEED)
    Xr = rng.standard_normal((len(cats), 50)).astype(np.float32)
    Xr /= (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-8)
    r_ari, r_ami = kmeans_ari_ami(Xr, cats, k=18)
    _, r_pur = knn_purity(Xr, cats, k=15)
    # analytic chance floor for the macro-averaged purity = 1/#families
    n_fam = len(np.unique(cats[cats != OTHER]))
    res = dict(n_labeled=n_lab, n_families=int(n_fam),
               real_f2llm=real,
               random_gaussian=dict(aRI=round(r_ari, 4), aMI=round(r_ami, 4), knn_purity=round(r_pur, 4)),
               analytic_chance_purity=round(1.0 / n_fam, 4), seed=SEED)
    (OUT / "random_control.json").write_text(json.dumps(res, indent=2))
    print("\n=== (2) Random-embedding negative control ===")
    print(f"  real f2llm : aRI={real['aRI']:.3f}  aMI={real['aMI']:.3f}  purity={real['knn_purity']:.3f}")
    print(f"  random     : aRI={r_ari:.3f}  aMI={r_ami:.3f}  purity={r_pur:.3f}  (chance purity={1.0/n_fam:.3f})")
    return res


# ---------------------------------------------------------------- (3) nearest neighbours
ANCHORS = ["pleural effusion", "pulmonary embolism", "hepatic steatosis",
           "coronary artery calcification", "splenomegaly", "abdominal aortic aneurysm",
           "pneumothorax", "bowel obstruction", "biliary ductal dilatation", "lymphadenopathy"]


def nn_examples(topk=8, chunk=20000):
    emb_path = ROOT / "_emb_f2llm_tmp.npy"
    concepts = np.load(ROOT / "concept_bank.f2llm_emb.npz", allow_pickle=True)["concepts"]
    cats = np.load(UMAP_DIR / "f2llm" / "categories.npy", allow_pickle=True)
    X = np.load(emb_path, mmap_mode="r")
    assert X.shape[0] == len(concepts) == len(cats), (X.shape, len(concepts), len(cats))
    lcs = np.array([str(c).lower() for c in concepts], dtype=object).astype(str)
    NEG = re.compile(r'\b(no|without|negative|absent|absence|not|unremarkable|resolved|ruled out|free of)\b')

    # pick the shortest POSITIVE (non-negated) concept containing each anchor term
    anchor_idx, used = [], []
    for term in ANCHORS:
        hits = list(np.where(np.char.find(lcs, term) >= 0)[0])
        if not hits:
            continue
        pos = [j for j in hits if not NEG.search(str(concepts[j]).lower())]
        pool = pos if pos else hits
        i = min(pool, key=lambda j: len(str(concepts[j])))
        anchor_idx.append(i); used.append(term)
    A = np.array(X[anchor_idx], dtype=np.float32)  # copy (X is a read-only memmap)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)

    sims = np.empty((X.shape[0], len(anchor_idx)), dtype=np.float32)
    t0 = time.time()
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        b = np.array(X[s:e], dtype=np.float32)  # copy (X is a read-only memmap)
        b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        sims[s:e] = b @ A.T
    print(f"\n=== (3) Nearest-neighbour clinical relationships  [{time.time()-t0:.0f}s over {X.shape[0]:,} concepts] ===")

    out = {}
    for col, (term, ai) in enumerate(zip(used, anchor_idx)):
        order = np.argsort(-sims[:, col])
        anchor_str = str(concepts[ai])
        nbrs, seen = [], {anchor_str.lower()}
        for j in order:
            if j == ai:
                continue
            s = str(concepts[j])
            if s.lower() in seen:
                continue
            seen.add(s.lower())
            nbrs.append(dict(concept=s, cosine=round(float(sims[j, col]), 4), family=str(cats[j])))
            if len(nbrs) >= topk:
                break
        same_fam = np.mean([n["family"] == str(cats[ai]) for n in nbrs]) if nbrs else 0.0
        out[term] = dict(anchor=anchor_str, anchor_family=str(cats[ai]),
                         self_sim_check=round(float(sims[ai, col]), 4),
                         neighbour_same_family_frac=round(float(same_fam), 3), neighbours=nbrs)
        print(f"\n  [{term}] anchor: \"{anchor_str}\"  (family={cats[ai]}, self-sim={sims[ai,col]:.3f})")
        for n in nbrs[:5]:
            print(f"      {n['cosine']:.3f}  {n['concept'][:80]}")
    (OUT / "nn_examples.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    t = time.time()
    cross_embedder()
    random_control()
    nn_examples()
    print(f"\nSaved to {OUT}  [total {time.time()-t:.0f}s]")

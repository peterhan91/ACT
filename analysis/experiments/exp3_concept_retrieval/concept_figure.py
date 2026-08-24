#!/usr/bin/env python3
"""
exp3 — concept-conditioned CT retrieval + most-attentive-slice localization.

A MONET-style qualitative figure (Kim et al., Nat Med 2024, Fig 2 — "images with
high concept-presence scores") for OUR 3D-CLIP, on the held-out PMBB pools from
exp1:
  * pmbb_chest_nc — 9,097 non-contrast chest CTs, 18 CT-RATE findings
  * pmbb_abd_ce   — 14,290 contrast abdomen CTs, 30 Merlin findings

For each finding we (a) retrieve the top-K CTs by the MONET concept-presence score
and (b) localize, within each retrieved CT, the most-attentive axial slice for that
finding. Layout = MONET Fig 2: rows = findings, each row = the K top CTs by their
most-attentive slice.

SCORING — MONET concept presence score (their Methods), reference-normalized and
prompt-ensembled (NOT raw cosine, NOT exp1's plain pos/"no"-neg softmax):
    p(x, f) = mean_t  sigmoid( (cos(x, Tc_t) - cos(x, Tr_t)) * s )
  where for each template t, Tc_t = "<template with finding f>", Tr_t = the same
  template WITHOUT the finding (the reference prompt that removes template bias),
  and s = the model's CLIP logit scale (1/temperature). The 2-way softmax of
  [cos_c, cos_r] equals sigmoid(cos_c - cos_r). x is a volume embedding (retrieval)
  or a per-slice embedding (localization) — the SAME score at both granularities.

"2D + 1D" — clip_3d_ctrate_merlin_v1 encodes each axial slice with a DINOv2 ViT-B
(2D) and fuses the slice sequence with a CLS transformer (1D). Volume embeddings
come from exp1's cached features (free). Per-slice embeddings come from a forward
hook on the ViT backbone, pushed through the SAME image projection head, then scored
identically. argmax slice = the most concept-present slice for that finding.

Caveat: the projection head was trained on the fused CLS token, so per-slice
projected scoring is a faithful but approximate saliency. Model = the exact exp1
`ours` checkpoint (crimson env), so figure and exp1 are consistent.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CLEAR3D_REPO", "/path/to/ACT/analysis")
os.environ.setdefault("CLIP3D_RUN_TAG", "v1")
os.environ.setdefault("CLIP3D_BEST", "clip_3d_ctrate_merlin_v1")
sys.path.insert(0, os.environ["CLEAR3D_REPO"])

HERE = Path(__file__).resolve().parent
OURS_RES = HERE.parent / "exp1_zeroshot" / "ours" / "results"
MINED_DIR = HERE.parent / "exp1_zeroshot" / "pmbb_labels" / "labels"
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

POOLS = {"chest": "pmbb_chest_nc", "abd": "pmbb_abd_ce"}
POOL2REGION = {v: k for k, v in POOLS.items()}
TOPK = 10
ROWS_PER_PAGE = 18
# The 160-slice volumes are air-padded; a near-empty slice's embedding can spuriously
# win the per-slice argmax. Only consider slices with real tissue: uint8>64 == HU>-500
# (soft tissue), and >=5% of the 224x224 slice must be tissue.
FG_THRESH = 64
FG_MIN = 0.05

# MONET-style (concept template, reference template) pairs. The reference is the
# template WITHOUT the finding — it removes template bias (MONET Methods). The bare
# finding name vs the generic modality is the exp1-anchored pair; the rest ensemble.
TEMPLATES = {
    "chest": [("{f}", "chest ct"),
              ("chest ct showing {f}", "chest ct showing"),
              ("axial ct image with {f}", "axial ct image")],
    "abd":   [("{f}", "abdominal ct"),
              ("abdominal ct showing {f}", "abdominal ct showing"),
              ("axial ct image with {f}", "axial ct image")],
}


def build_text(region, prompts):
    """-> (Cc (L,P,d), Cr (P,d)) normalized text embeddings for concept & reference."""
    from clear3d.features import encode_texts_clip
    tpl = TEMPLATES[region]
    P = len(tpl)
    concept_strs, ref_strs = [], [r for (_c, r) in tpl]
    for f in prompts:                                   # finding-major order
        for (cfmt, _r) in tpl:
            concept_strs.append(cfmt.format(f=f))
    Cc = encode_texts_clip(concept_strs).reshape(len(prompts), P, -1)   # (L,P,d)
    Cr = encode_texts_clip(ref_strs)                                    # (P,d)
    return Cc, Cr


def presence(feats, Cc, Cr, s):
    """MONET concept-presence. feats (M,d) torch; Cc (L,P,d); Cr (P,d). -> (M,L)."""
    import torch
    cc = torch.einsum("md,lpd->mlp", feats, Cc)            # (M,L,P)
    cr = torch.einsum("md,pd->mp", feats, Cr)              # (M,P)
    return torch.sigmoid((cc - cr.unsqueeze(1)) * s).mean(dim=2)   # (M,L)


_MINED = {}


def mined_val(pool, vn, lab):
    m = _MINED[pool]
    if vn in m.index and lab in m.columns:
        try:
            return int(m.loc[vn, lab])
        except Exception:
            return -1
    return -1


def run_pool(region, pool, model):
    import torch
    from clear3d.features import encode_images
    from clear3d.paths import DATASETS
    import h5py

    s = float(model.logit_scale.exp().item())             # CLIP inv-temperature
    meta = json.load(open(OURS_RES / f"{pool}__plain__results.json"))
    labels = meta["labels"]
    prompts = [l.lower() for l in labels]
    from clear3d.data import load_split
    df = load_split(pool).reset_index(drop=True)
    _MINED[pool] = pd.read_csv(MINED_DIR / f"{pool}_labels.csv").set_index("VolumeName")

    # ---- text (concept + reference, ensembled) ----
    Cc_np, Cr_np = build_text(region, prompts)
    Cc = torch.tensor(Cc_np, device="cuda")
    Cr = torch.tensor(Cr_np, device="cuda")

    # ---- volume retrieval: MONET presence on exp1's cached volume embeddings ----
    feats = encode_images(pool)                            # (N,d) normalized, cached
    assert len(df) == len(feats), f"{pool}: df {len(df)} != feats {len(feats)}"
    vol = torch.tensor(feats, device="cuda")
    vpres = presence(vol, Cc, Cr, s).cpu().numpy()        # (N,L)
    topk_rows = np.argsort(-vpres, axis=0)[:TOPK]         # (TOPK,L)

    # ---- slice localization: GRADIENT ATTRIBUTION through fusion+projection ----
    # The projection head was trained on the FUSED CLS token, so scoring an isolated
    # slice token with it is out-of-distribution. Instead we attribute the volume's
    # concept margin back to each slice THROUGH the real CLS-fusion transformer
    # (grad-CAM on the slice axis): saliency_i = relu( sum_d grad_i · slice_i ), where
    # grad = d(margin)/d(slice features). Backbone stays no-grad (cheap); only the
    # 4-layer fusion + projection are differentiated. argmax = the slice that most
    # increases this finding's presence in the ACTUAL volume embedding.
    visual = model.visual
    assert hasattr(visual, "backbone") and hasattr(visual, "slice_fusion")
    cls_tok = visual.cls_token                            # (1,1,dim)
    cap = {}
    handle = visual.backbone.register_forward_hook(lambda _m, _i, o: cap.__setitem__("sf", o))
    h5 = h5py.File(DATASETS[pool]["h5"], "r")["ct_volumes"]
    needed = sorted({int(r) for r in topk_rows.flatten()})
    print(f"[{pool}] {len(labels)} findings × top-{TOPK} -> {len(needed)} unique volumes", flush=True)
    sf_store, fg_frac = {}, {}
    for k, row in enumerate(needed):
        v = np.asarray(h5[int(df.iloc[row]["h5_idx"])])    # (160,224,224) uint8
        fg_frac[row] = (v.reshape(v.shape[0], -1) > FG_THRESH).mean(axis=1)
        x = torch.from_numpy(np.repeat(v[None], 3, 0)[None]).float().cuda()
        cap.clear()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = visual(x)
        sf_store[row] = cap["sf"].detach()                 # (D,dim) bf16 — slice features
        if (k + 1) % 50 == 0:
            print(f"  [{pool}] encoded {k + 1}/{len(needed)}", flush=True)
    handle.remove()

    records = []
    for j, lab in enumerate(labels):
        Ccj = Cc[j]                                        # (P,d)
        for rank, row in enumerate(topk_rows[:, j]):
            row = int(row)
            sf = sf_store[row].clone().float().requires_grad_(True)   # (D,dim) fp32 leaf
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                tokens = torch.cat([sf.unsqueeze(0), cls_tok.float()], dim=1)   # (1,D+1,dim)
                fused = visual.slice_fusion(tokens)
                vemb = torch.nn.functional.normalize(visual.projection(fused[:, -1]).float(), dim=-1)
            margin = (vemb @ Ccj.t() - vemb @ Cr.t()).mean()           # concept presence margin
            g = torch.autograd.grad(margin, sf)[0]                     # (D,dim)
            sal = torch.relu((g * sf).sum(-1)).detach().cpu().numpy()  # (D,) grad-CAM
            sal[fg_frac[row] < FG_MIN] = -np.inf                       # ignore air-padded slices
            sidx = int(np.argmax(sal))
            vn = str(df.iloc[row]["VolumeName"])
            records.append(dict(
                region=region, pool=pool, finding=lab, rank=rank, row=row,
                h5_idx=int(df.iloc[row]["h5_idx"]), VolumeName=vn,
                vol_score=float(vpres[row, j]), slice_idx=sidx,
                slice_score=float(sal[sidx] if np.isfinite(sal[sidx]) else 0.0),
                mined=mined_val(pool, vn, lab)))
    render(region, pool, records)
    return records


def render(region, pool, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import h5py
    from clear3d.paths import DATASETS

    rec_df = pd.DataFrame(records)
    rec_df.to_csv(OUT / f"manifest_{region}.csv", index=False)
    labels = list(dict.fromkeys(rec_df["finding"]))
    h5 = h5py.File(DATASETS[pool]["h5"], "r")["ct_volumes"]
    pages = [labels[i:i + ROWS_PER_PAGE] for i in range(0, len(labels), ROWS_PER_PAGE)]
    for pi, page in enumerate(pages, 1):
        nrows = len(page)
        fig, axes = plt.subplots(nrows, TOPK, figsize=(TOPK * 1.5, nrows * 1.6), squeeze=False)
        for ri, lab in enumerate(page):
            sub = rec_df[rec_df["finding"] == lab].sort_values("rank")
            for ci, (_, r) in enumerate(sub.iterrows()):
                ax = axes[ri][ci]
                ax.imshow(np.asarray(h5[int(r["h5_idx"]), int(r["slice_idx"])]),
                          cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([]); ax.set_yticks([])
                tag = {1: "+", 0: "-"}.get(int(r["mined"]), "?")
                ax.set_title(f"{r['vol_score']:.2f}{tag} s{int(r['slice_idx'])}", fontsize=6, pad=1)
                if ci == 0:
                    ax.set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center")
        fig.suptitle(f"Ours 3D-CLIP — MONET-style concept retrieval, PMBB {region} "
                     f"(top-{TOPK} CTs/finding, most-attentive slice; cell = presence, "
                     f"+/- = mined label, s# = slice idx)", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        suffix = f"_p{pi}" if len(pages) > 1 else ""
        out = OUT / f"concept_{region}{suffix}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"[{pool}] wrote {out}  ({nrows} findings)", flush=True)


def main():
    from clear3d.features import load_3d_clip
    model = load_3d_clip()
    allr = []
    for region, pool in POOLS.items():
        print(f"\n===== {pool} ({region}) =====", flush=True)
        allr.extend(run_pool(region, pool, model))
    pd.DataFrame(allr).to_csv(OUT / "manifest_all.csv", index=False)
    print(f"\n[done] {len(allr)} cells -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

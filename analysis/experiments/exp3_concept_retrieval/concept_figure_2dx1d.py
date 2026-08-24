#!/usr/bin/env python3
"""
exp3 (variant) — slice selection by 2D x 1D attention (max-attention slice).

Instead of choosing the slice from the 1D slice-aggregation gradient alone
(concept_figure.py), here we compute the FULL 3D concept grad-CAM — attribute the
volume's concept margin back to every (slice, patch) at once — and pick the slice
that contains the global maximum of that combined 2D x 1D attention. The 1D factor
(which slice the fusion uses) and the 2D factor (in-plane patch response) come out
of one attribution; the same map is the in-plane overlay, so selection and overlay
are consistent.

Efficient: per volume, run the backbone up to block -2 once (no-grad, cached);
per finding, differentiate only the last 2 blocks + fusion. (Verified: the last-2-
block replication reproduces the model slice embedding at cos 0.99999.)

Honest finding (see out/_slice_criteria.png): on this localization-untrained model
the 2D x 1D selection fixes findings in large organs (liver/heart) but can miss
focal ones (a lung nodule whose in-plane peak lands off-target). Kept ALONGSIDE the
1D figures, not as a replacement.

Outputs: out/concept_<region>_2dx1d.png (slice montage) +
         out/concept_<region>_2dx1d_overlay.png (with the grad-CAM).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CLEAR3D_REPO", "/path/to/ACT/analysis")
os.environ.setdefault("CLIP3D_RUN_TAG", "v1")
os.environ.setdefault("CLIP3D_BEST", "clip_3d_ctrate_merlin_v1")
sys.path.insert(0, os.environ["CLEAR3D_REPO"])

import concept_figure as CF

OUT = CF.OUT
TOPK = CF.TOPK
ROWS_PER_PAGE = CF.ROWS_PER_PAGE
FG_THRESH, FG_MIN = CF.FG_THRESH, CF.FG_MIN


def main():
    import torch, h5py
    from scipy.ndimage import gaussian_filter, zoom
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from clear3d.features import load_3d_clip, encode_texts_clip
    from clear3d.paths import DATASETS

    model = load_3d_clip(); visual = model.visual
    bb, cls = visual.backbone, visual.cls_token
    MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def preprocess(vol):
        x = torch.from_numpy(np.repeat(vol[:, None], 3, 1)).float().cuda()
        xf = x.reshape(x.shape[0], -1)
        mn, mx = xf.min(1, keepdim=True)[0], xf.max(1, keepdim=True)[0]
        rng = torch.where((mx - mn) < 1e-5, torch.ones_like(mx), mx - mn)
        return (((xf - mn) / rng).reshape(x.shape[0], 3, 224, 224) - MEAN) / STD

    def vol_h(vol):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            t = bb.prepare_tokens_with_masks(preprocess(vol))
            for blk in bb.blocks[:-2]:
                t = blk(t)
        return t.detach()

    def cam3d(h, Ccj, Cr):
        a = bb.blocks[-2](h)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            clt = bb.norm(bb.blocks[-1](a))[:, 0]
            v = torch.nn.functional.normalize(visual.projection(
                visual.slice_fusion(torch.cat([clt.unsqueeze(0), cls.float()], 1))[:, -1]).float(), dim=-1)
        margin = (v @ Ccj.t() - v @ Cr.t()).mean()
        g = torch.autograd.grad(margin, a)[0]
        return torch.relu((g[:, 5:].float() * a[:, 5:].float()).sum(-1)).reshape(-1, 16, 16).detach().cpu().numpy()

    man = pd.read_csv(OUT / "manifest_all.csv")
    for region, pool in CF.POOLS.items():
        sub = man[man["pool"] == pool].copy()
        labels = list(dict.fromkeys(sub["finding"]))
        prompts = [l.lower() for l in labels]
        tpl = CF.TEMPLATES[region]
        Cc = torch.tensor(encode_texts_clip([c.format(f=f) for f in prompts for c, _ in tpl]),
                          device="cuda").reshape(len(prompts), len(tpl), -1)
        Cr = torch.tensor(encode_texts_clip([r for _, r in tpl]), device="cuda")
        jmap = {lab: j for j, lab in enumerate(labels)}
        h5 = h5py.File(DATASETS[pool]["h5"], "r")["ct_volumes"]

        slc, cam = {}, {}
        for h5i, grp in sub.groupby("h5_idx"):
            vol = np.asarray(h5[int(h5i)])
            fg = (vol.reshape(vol.shape[0], -1) > FG_THRESH).mean(1)
            h = vol_h(vol)
            for _, r in grp.iterrows():
                c = cam3d(h, Cc[jmap[r["finding"]]], Cr)            # (160,16,16)
                peak = c.reshape(c.shape[0], -1).max(1).copy()
                peak[fg < FG_MIN] = -1
                k = int(peak.argmax())                              # max-attention slice
                key = (r["finding"], int(r["rank"]))
                slc[key] = (vol[k], k)
                cm = gaussian_filter(zoom(c[k], 224 / 16, order=1), sigma=4)
                cam[key] = cm / (cm.max() + 1e-8)
        print(f"[{pool}] computed {len(slc)} 2Dx1D slices", flush=True)

        for overlay in (False, True):
            pages = [labels[i:i + ROWS_PER_PAGE] for i in range(0, len(labels), ROWS_PER_PAGE)]
            for pi, page in enumerate(pages, 1):
                fig, axes = plt.subplots(len(page), TOPK, figsize=(TOPK * 1.5, len(page) * 1.6), squeeze=False)
                for ri, lab in enumerate(page):
                    for ci in range(TOPK):
                        ax = axes[ri][ci]; key = (lab, ci)
                        if key in slc:
                            img, k = slc[key]
                            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                            if overlay:
                                cm = cam[key]
                                ax.imshow(cm, cmap="turbo", alpha=np.clip(cm ** 0.6, 0, 1) * 0.85, vmin=0, vmax=1)
                            ax.set_title(f"s{k}", fontsize=6, pad=1)
                        ax.set_xticks([]); ax.set_yticks([])
                        if ci == 0:
                            ax.set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center")
                tag = "overlay" if overlay else "slice"
                fig.suptitle(f"Ours 3D-CLIP — PMBB {region}: 2Dx1D max-attention slice ({tag}). "
                             f"Slice = argmax over (slice,patch) of concept grad-CAM.", fontsize=9)
                fig.tight_layout(rect=[0, 0, 1, 0.98])
                suf = f"_p{pi}" if len(pages) > 1 else ""
                name = f"concept_{region}_2dx1d{'_overlay' if overlay else ''}{suf}.png"
                fig.savefig(OUT / name, dpi=130, bbox_inches="tight"); plt.close(fig)
                print(f"[{pool}] wrote {OUT / name}", flush=True)
    print("[done 2dx1d]", flush=True)


if __name__ == "__main__":
    main()

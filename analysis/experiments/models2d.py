#!/usr/bin/env python3
"""
Shared 2D-baseline encoders for applying 2D image-text models to CT volumes.

Models: MedSigLIP (google/medsiglip-448), BiomedCLIP (open_clip), OpenAI CLIP
(openai/clip-vit-base-patch16). Each is a 2D model; we encode every axial slice
through the model's OWN official image pipeline (its processor: resize +
normalize) and AVERAGE-POOL slice embeddings into a volume embedding. Text uses
the model's own tokenizer + text tower. Embeddings are L2-normalized so cosine =
dot — works for both exp1 zero-shot classification (softmax_pos_neg) and exp2
retrieval.

Per the project rule "for 2D baselines we use THEIR data pipeline": the model's
processor (resize/normalize) is applied unchanged. The CT->RGB conversion follows
Google's official MedGemma-1.5 high-dimensional-CT representation (the closest
official CT recipe for this SigLIP-lineage encoder): three fixed HU windows packed
as R/G/B — wide (-1024,1024) / soft-tissue (-135,215) / narrow (0,80) — with
uniform slice sampling (~85/volume). Both are overridable via CT2D_WINDOWS /
CT2D_NSLICES for ablation. Volumes are read from the ORIGINAL native scans
(native.py); the h5 path is kept only as a controlled ablation (--source h5).

Env: `ml311` (open_clip + transformers + timm, torch+cu129 on the GH200).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = Path(__file__).resolve().parent / "_cache2d"     # shared by exp1 + exp2

CONFIG = {
    "medsiglip": "google/medsiglip-448",
    "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "openai_clip": "openai/clip-vit-base-patch16",
}


def uint8_to_hu(vol_u8: np.ndarray) -> np.ndarray:
    return vol_u8.astype(np.float32) / 255.0 * 2000.0 - 1000.0


# Google's official CT->RGB representation (MedGemma 1.5 high-dimensional-CT
# notebook): three fixed HU windows packed into R/G/B channels. Region-agnostic.
_WINDOW_SETS = {
    "3win":       [(-1024.0, 1024.0), (-135.0, 215.0), (0.0, 80.0)],  # wide / soft-tissue / narrow
    "softtissue": [(-160.0, 240.0)] * 3,                              # single soft-tissue (ablation)
    "global":     [(-1000.0, 1000.0)] * 3,                            # old global window (ablation)
}
import os as _os  # noqa: E402
WINDOW_NAME = _os.environ.get("CT2D_WINDOWS", "3win")
CT_WINDOWS = _WINDOW_SETS.get(WINDOW_NAME, _WINDOW_SETS["3win"])
N_SLICES = int(_os.environ.get("CT2D_NSLICES", "85"))   # uniform slices/volume; 0 = all


def _win(hu, lo, hi):
    x = np.clip(hu.astype(np.float32), lo, hi); x -= lo; x /= (hi - lo); return x * 255.0


def window_rgb(vol_hu: np.ndarray) -> np.ndarray:
    """(D,H,W) HU -> (D,H,W,3) uint8 via the three CT_WINDOWS as R/G/B."""
    return np.stack([_win(vol_hu, lo, hi) for lo, hi in CT_WINDOWS], -1).round().astype(np.uint8)


def _sample_idx(D: int, n: int):
    """Uniform slice indices (Google notebook formula); n<=0 or D<=n -> all slices."""
    if n <= 0 or D <= n:
        return list(range(D))
    return [int(round(i / n * (D - 1))) for i in range(1, n + 1)]


def _to_pil(slices_rgb: np.ndarray):
    """(b,H,W,3) uint8 -> list of RGB PIL images."""
    return [Image.fromarray(s) for s in slices_rgb]


class _Base:
    name = "base"
    prompt_template = "{}"   # per-model official caption template (overridden below)

    def __init__(self, device: str = DEVICE):
        self.device = device

    # subclasses implement:
    #   _encode_image_slices(u8 (b,H,W) uint8) -> (b,d) tensor on device
    #   _encode_text(list[str])               -> (b,d) tensor on device

    @torch.no_grad()
    def encode_volumes(self, volumes_hu, slice_bs: int = 64) -> np.ndarray:
        """Iterable of (D,H,W) HU volumes -> (N, d) L2-normalized volume embeddings:
        uniform-sample slices, 3-window RGB, per-slice encode through the model's own
        pipeline, then average-pool."""
        slice_bs = getattr(self, "slice_bs", None) or slice_bs
        out = []
        for k, vol in enumerate(volumes_hu):
            vol = np.asarray(vol)
            rgb = window_rgb(vol[_sample_idx(vol.shape[0], N_SLICES)])   # (S,H,W,3) uint8
            embs = [self._encode_image_slices(rgb[i:i + slice_bs])
                    for i in range(0, len(rgb), slice_bs)]
            v = torch.cat(embs, 0).mean(0)
            out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
            if (k + 1) % 200 == 0:
                print(f"  [{self.name}] encoded {k + 1} volumes", flush=True)
        return np.stack(out, 0)

    @torch.no_grad()
    def encode_texts(self, texts, bs: int = 64) -> np.ndarray:
        out = [self._encode_text(list(texts[i:i + bs])) for i in range(0, len(texts), bs)]
        e = torch.cat(out, 0)
        return torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()


class OpenAICLIP(_Base):
    name = "openai_clip"
    prompt_template = "a photo of a {}"          # OpenAI CLIP's standard zero-shot template

    def __init__(self, device: str = DEVICE):
        super().__init__(device)
        from transformers import CLIPModel, CLIPProcessor
        cid = CONFIG[self.name]
        self.m = CLIPModel.from_pretrained(cid).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(cid)

    def _encode_image_slices(self, u8):
        px = self.proc(images=_to_pil(u8), return_tensors="pt").pixel_values.to(self.device)
        return self.m.get_image_features(pixel_values=px)

    def _encode_text(self, texts):
        t = self.proc(text=texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=77).to(self.device)
        return self.m.get_text_features(input_ids=t.input_ids, attention_mask=t.attention_mask)


class MedSigLIP(_Base):
    name = "medsiglip"
    prompt_template = "a photo of {}"    # MedSigLIP card captions are "a photo of ..." form
    slice_bs = 64        # 448px tower is heavy; a smaller slice batch keeps GPU mem modest

    def __init__(self, device: str = DEVICE):
        super().__init__(device)
        from transformers import AutoModel, AutoProcessor
        cid = CONFIG[self.name]
        # bf16: the cost is the 448px SigLIP vision tower over every axial slice
        # (~8 s/vol in fp32 under GPU contention; use_fast made no difference). bf16
        # ~halves compute+memory at negligible zero-shot cost.
        self.m = AutoModel.from_pretrained(cid, torch_dtype=torch.bfloat16).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(cid)

    def _encode_image_slices(self, u8):
        px = self.proc(images=_to_pil(u8), return_tensors="pt").pixel_values.to(
            self.device, dtype=torch.bfloat16)
        return self.m.get_image_features(pixel_values=px).float()

    def _encode_text(self, texts):
        # SigLIP: pad to its fixed 64-token context; no attention mask needed.
        t = self.proc(text=texts, return_tensors="pt", padding="max_length",
                      max_length=64, truncation=True).to(self.device)
        return self.m.get_text_features(input_ids=t.input_ids).float()


class BiomedCLIP(_Base):
    name = "biomedclip"
    prompt_template = "this is a photo of {}"   # BiomedCLIP example-notebook template
    ctx = 256

    def __init__(self, device: str = DEVICE):
        super().__init__(device)
        from open_clip import create_model_from_pretrained, get_tokenizer
        cid = CONFIG[self.name]
        self.m, self.preprocess = create_model_from_pretrained(cid)
        self.m = self.m.to(device).eval()
        self.tok = get_tokenizer(cid)

    def _encode_image_slices(self, u8):
        imgs = torch.stack([self.preprocess(im) for im in _to_pil(u8)]).to(self.device)
        return self.m.encode_image(imgs)

    def _encode_text(self, texts):
        toks = self.tok(texts, context_length=self.ctx).to(self.device)
        return self.m.encode_text(toks)


_REGISTRY = {"medsiglip": MedSigLIP, "biomedclip": BiomedCLIP, "openai_clip": OpenAICLIP}


def load(name: str, device: str = DEVICE) -> _Base:
    return _REGISTRY[name](device)


def inspect_native_feats(enc: _Base, model_name: str, manifest_df, cache_dir=None,
                         cache_name=None) -> np.ndarray:
    """2D image feats for the INSPECT phenotype eval from NATIVE CTPA NIfTIs (the
    2D models' 'original pipeline' = per-axial-slice processor + avg-pool; the only
    fixed step is the HU->image window inside encode_volumes). Each NIfTI is read,
    oriented to canonical RAS, transposed to (D=Z, H=Y, W=X) axial. Cached.

    `cache_name` overrides the cache filename (used for sharded parallel runs, where
    each process encodes a contiguous block of `manifest_df` to its own shard file)."""
    import native as NV   # robust raw-CT loader (4D-collapse + affine-sanitize), as 3D path uses
    from nibabel.orientations import (io_orientation, axcodes2ornt,
                                      ornt_transform, apply_orientation)
    cdir = Path(cache_dir) if cache_dir else CACHE
    cdir.mkdir(exist_ok=True)
    p = cdir / (cache_name or f"{model_name}__inspect_pheno__imgfeat.npy")
    if p.exists():
        cached = np.load(p)
        if len(cached) == len(manifest_df):
            print(f"[cache] reuse {p}")
            return cached
        print(f"[cache] STALE {p} ({len(cached)} != {len(manifest_df)}) -> recompute", flush=True)

    ras_ornt = axcodes2ornt(("R", "A", "S"))

    def _volumes():
        for pth in manifest_df["Path"]:
            # load_hu_affine collapses 4D frames + sanitizes degenerate affines
            # (the same robust path the 3D baselines use); then reorient the clean
            # 3D array to canonical RAS so axis 2 is the axial (S) slice axis.
            hu, aff = NV.load_hu_affine({"kind": "nii", "path": str(pth)})  # (X,Y,Z) HU
            t = ornt_transform(io_orientation(aff), ras_ornt)
            ras = apply_orientation(hu.astype(np.float32), t)              # (R,A,S)
            yield np.ascontiguousarray(np.transpose(ras, (2, 1, 0)))       # (S,A,R)=(D,H,W)

    img = enc.encode_volumes(_volumes())
    np.save(p, img)
    print(f"[cache] saved {p} {img.shape}")
    return img


def dataset_image_feats(enc: _Base, name: str, dataset: str, h5_path: str, h5_idx) -> np.ndarray:
    """Per-volume 2D-encoded image features, cached at CLEAR_CT_EXPS/_cache2d/
    (shared exp1<->exp2: CT-RATE slices + model are identical across both)."""
    import h5py
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{name}__{dataset}__imgfeat.npy"
    if p.exists():
        cached = np.load(p)
        if len(cached) == len(h5_idx):
            print(f"[cache] reuse {p}")
            return cached
        print(f"[cache] STALE {p} ({len(cached)} rows != {len(h5_idx)} requested) "
              f"-> recompute", flush=True)
    f = h5py.File(h5_path, "r")["ct_volumes"]
    vols = (uint8_to_hu(np.asarray(f[int(j)])) for j in h5_idx)
    img = enc.encode_volumes(vols)
    np.save(p, img)
    print(f"[cache] saved {p} {img.shape}")
    return img


def native_volume_feats(enc: _Base, model_name: str, dataset: str,
                        cache_dir=None, limit: int = 0) -> np.ndarray:
    """Per-volume 2D feats from the ORIGINAL native scans (CT-RATE/RSNA/PMBB NIfTI,
    RAD-ChestCT .npz) via native.py — the faithful 'official pipeline' source for the
    2D baselines (full-resolution slices, not our 224^3 h5). Reorients to canonical
    RAS so axis 0 is axial; windowing + slice sampling happen in encode_volumes.
    Cached at _cache2d/<model>__<dataset>__native_imgfeat.npy. Order follows
    native.native_index == clear3d.data.load_split, so it aligns with the y-matrix."""
    import native as NV
    from nibabel.orientations import (io_orientation, axcodes2ornt,
                                      ornt_transform, apply_orientation)
    _df, items = NV.native_index(dataset)
    if limit:
        items = items[:limit]
    cdir = Path(cache_dir) if cache_dir else CACHE
    cdir.mkdir(exist_ok=True)
    p = cdir / f"{model_name}__{dataset}__native_imgfeat.npy"
    if p.exists():
        cached = np.load(p)
        if len(cached) == len(items):
            print(f"[cache] reuse {p}")
            return cached
        print(f"[cache] STALE {p} ({len(cached)} != {len(items)}) -> recompute", flush=True)
    ras = axcodes2ornt(("R", "A", "S"))

    def _vols():
        for it in items:
            hu, aff = NV.load_hu_affine(it)                          # (X,Y,Z) HU
            t = ornt_transform(io_orientation(aff), ras)
            r = apply_orientation(hu.astype(np.float32), t)          # (R,A,S)
            yield np.ascontiguousarray(np.transpose(r, (2, 1, 0)))   # (S,A,R)=(D,H,W)

    img = enc.encode_volumes(_vols())
    np.save(p, img)
    print(f"[cache] saved {p} {img.shape}")
    return img

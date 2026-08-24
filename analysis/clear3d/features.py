"""
3D-CLIP loading + image / text feature extraction with on-disk caching.

The 3D-CLIP construction logic (handles transformer fusion / wider text tower)
is the same as `clip_3d_eval/eval_all.py`'s `build_and_load`; we import it
directly to avoid duplication.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .paths import (
    CLIP3D_BEST, CLIP3D_CONFIGS, CLIP3D_REPO, EVAL_REPO,
    DATASETS, cache_img_feats,
)
from .data import CTH5Dataset, load_split

# The eval repo's build_and_load handles the model construction quirks.
sys.path.insert(0, CLIP3D_REPO)
sys.path.insert(0, EVAL_REPO)
import clip                       # noqa: E402
from eval_all import build_and_load  # noqa: E402


_MODEL_CACHE = {"name": None, "model": None}


def load_3d_clip(name: str = CLIP3D_BEST, context_length: int = 77):
    """Load (and memoize) a 3D-CLIP checkpoint by config name."""
    if _MODEL_CACHE["name"] == name and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"]
    with open(CLIP3D_CONFIGS) as f:
        cfgs = json.load(f)
    cfg = cfgs[name]
    model = build_and_load(cfg, context_length=context_length)
    model = model.to("cuda").eval()
    _MODEL_CACHE["name"] = name
    _MODEL_CACHE["model"] = model
    return model


def free_3d_clip():
    """Drop the cached 3D-CLIP from GPU; the SFR/Harrier load is heavy."""
    if _MODEL_CACHE["model"] is not None:
        _MODEL_CACHE["model"] = None
        _MODEL_CACHE["name"] = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def encode_images(dataset: str, *, batch_size: int = 4, num_workers: int = 4,
                  overwrite: bool = False) -> np.ndarray:
    """Encode every aligned volume in `dataset` with 3D-CLIP. Caches to disk."""
    out = cache_img_feats(dataset)
    if out.exists() and not overwrite:
        feats = np.load(out)
        print(f"[features:{dataset}] loaded cached img_feats {feats.shape}")
        return feats

    df = load_split(dataset)
    h5_indices = df["h5_idx"].values
    print(f"[features:{dataset}] encoding {len(h5_indices)} volumes from "
          f"{DATASETS[dataset]['h5']}...")
    t0 = time.time()
    model = load_3d_clip()
    ds = CTH5Dataset(DATASETS[dataset]["h5"], h5_indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats = []
    for batch in tqdm(loader, desc=f"img_feats {dataset}"):
        imgs = batch["img"].to("cuda", non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            f = model.encode_image(imgs)
        f = f.float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    np.save(out, feats)
    print(f"[features:{dataset}] saved {feats.shape} → {out}  ({time.time()-t0:.0f}s)")
    return feats


@torch.no_grad()
def encode_texts_clip(texts, context_length: int = 77, batch_size: int = 256) -> np.ndarray:
    """Encode a list of texts with the 3D-CLIP text tower (loaded if needed)."""
    model = load_3d_clip(context_length=context_length)
    from clip import _tokenizer  # noqa: E402  (uses package-side BPE)
    max_body = context_length - 2
    safe = []
    for t in texts:
        toks = _tokenizer.encode(t)
        if len(toks) > max_body:
            t = _tokenizer.decode(toks[:max_body])
        safe.append(t)

    out = []
    for i in range(0, len(safe), batch_size):
        chunk = safe[i:i + batch_size]
        tok = clip.tokenize(chunk, context_length).to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            f = model.encode_text(tok)
        f = f.float()
        f = f / f.norm(dim=-1, keepdim=True)
        out.append(f.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)

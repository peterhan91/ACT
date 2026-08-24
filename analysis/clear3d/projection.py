"""
The CLEAR projection: image features → concept-similarity → LLM-space embedding.

Mirrors `CLEAR.src.clear.pipeline.ZeroShotPipeline.project_to_concept_space`,
but lets us load the concept_bank lazily (it is large) and cache the projected
representation per (dataset, llm).

Convention:
    img_feats    : (N, 768)    float32, L2-normalized (3D-CLIP space)
    clip_text_emb: (M, 768)    float32, L2-normalized (3D-CLIP space)
    llm_emb      : (M, D_llm)  float32, L2-normalized (LLM space)

Output `llm_repr`: (N, D_llm) float32, L2-normalized.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .paths import CONCEPT_BANK_NPZ, cache_llm_repr


def _load_npz(path: Path) -> Tuple[np.ndarray, list]:
    z = np.load(path, allow_pickle=True)
    return z["emb"].astype(np.float32, copy=False), list(z["concepts"])


def load_concept_bank(*, llm: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load `clip_text_emb` and the requested `<llm>_emb`.

    `llm` is one of {"sfr", "harrier", "openai", "f2llm", "gteqwen2"}. Concept
    lists are checked for equality
    (they are written from the same sorted dedup list in get_embed_ct.py, but
    we sanity-check anyway).
    """
    clip_emb, concepts = _load_npz(CONCEPT_BANK_NPZ["clip_text"])
    llm_emb, concepts_llm = _load_npz(CONCEPT_BANK_NPZ[llm])
    if concepts != concepts_llm:
        raise ValueError(f"Concept lists in clip_text vs {llm} don't match.")

    out = {
        "concepts": concepts,
        "clip_text_emb": torch.from_numpy(clip_emb).to(device),
        "llm_emb": torch.from_numpy(llm_emb).to(device),
    }
    # The npz are NOT necessarily L2-normalized; explicitly normalize once.
    out["clip_text_emb"] = F.normalize(out["clip_text_emb"], dim=-1)
    out["llm_emb"] = F.normalize(out["llm_emb"], dim=-1)
    return out


@torch.no_grad()
def project_to_llm(
    img_feats: np.ndarray,
    *,
    llm: str,
    bank: Optional[Dict[str, torch.Tensor]] = None,
    img_chunk: int = 256,
    concept_chunk: int = 16384,
    device: str = "cuda",
) -> np.ndarray:
    """img_feats (N, 768) → llm_repr (N, D_llm), normalized.

    The two large tensors live on CPU (or `device`); we walk concept-axis chunks
    to keep peak memory bounded — same pattern as CLEAR's pipeline.
    """
    if bank is None:
        bank = load_concept_bank(llm=llm, device="cpu")
    clip_emb = bank["clip_text_emb"]   # (M, 768)
    llm_emb = bank["llm_emb"]          # (M, D)
    M, D = llm_emb.shape

    img = torch.from_numpy(img_feats).to(device)
    img = F.normalize(img, dim=-1)

    out = torch.zeros((img.shape[0], D), dtype=torch.float32, device=device)
    n_concept_chunks = (M + concept_chunk - 1) // concept_chunk
    for s in tqdm(range(0, M, concept_chunk), total=n_concept_chunks,
                  desc=f"project[{llm}]"):
        e = min(s + concept_chunk, M)
        ce = clip_emb[s:e].to(device)
        le = llm_emb[s:e].to(device)
        # (N, e-s) = img @ ce.T   ->   accumulate (N, D) += sims @ le
        sims = img @ ce.T
        out += sims @ le
    out = F.normalize(out, dim=-1)
    return out.cpu().numpy().astype(np.float32)


def get_or_compute_llm_repr(
    dataset: str, llm: str, img_feats: np.ndarray,
    *, overwrite: bool = False,
) -> np.ndarray:
    """Cached llm_repr per (dataset, llm)."""
    out = cache_llm_repr(dataset, llm)
    if out.exists() and not overwrite:
        repr_ = np.load(out)
        print(f"[projection:{dataset}/{llm}] cached llm_repr {repr_.shape}")
        return repr_
    repr_ = project_to_llm(img_feats, llm=llm)
    np.save(out, repr_)
    print(f"[projection:{dataset}/{llm}] saved {repr_.shape} → {out}")
    return repr_

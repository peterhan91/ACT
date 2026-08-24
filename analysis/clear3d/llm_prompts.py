"""
Embed positive / negative prompts for each label using the same LLM that the
concept bank was embedded with. Cached on disk so we never reload a 7-27B
encoder twice.

CLEAR uses SFR-Embedding-Mistral with the "Instruct: ...\\nQuery: ..." format
(see CLEAR's get_embed.py, get_detailed_instruct). Concept-bank embeddings in
get_embed_ct.py call SentenceTransformer.encode without that wrapper, which
applies whatever prompt is configured on the model card. To stay maximally
consistent, we use the *same call signature* as get_embed_ct.py here.
"""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .paths import LABELS_18, cache_label_emb


_LLM_REPO = {
    "sfr": "Salesforce/SFR-Embedding-Mistral",
    "harrier": "microsoft/harrier-oss-v1-27b",
    "openai": "text-embedding-3-large",
    "f2llm": "codefuse-ai/F2LLM-v2-14B",
    "gteqwen2": "Alibaba-NLP/gte-Qwen2-7B-instruct",
}

# Encoder dtype per backend (used by both the ST path and the AutoModel path).
_LLM_DTYPE = {
    "sfr": torch.bfloat16,
    "harrier": "auto",
    "f2llm": torch.bfloat16,
    "gteqwen2": torch.bfloat16,
}

# --- SentenceTransformer path (sfr / harrier; CLEAR's original recipe) ---
_LLM_MODEL_KWARGS: dict = {}
_LLM_TRUST_REMOTE: dict = {}          # default True (custom modeling)
_DOC_MODE: set = set()


def llm_st_cfg(llm: str):
    """(repo, dtype, model_kwargs, document_mode, trust_remote_code) for a
    SentenceTransformer backend (sfr / harrier)."""
    return (_LLM_REPO[llm], _LLM_DTYPE.get(llm, "auto"),
            _LLM_MODEL_KWARGS.get(llm, {}), llm in _DOC_MODE,
            _LLM_TRUST_REMOTE.get(llm, True))


# --- Official pure-transformers AutoModel path (f2llm / gteqwen2) ---
# Reproduces the model-card snippets EXACTLY: AutoModel + last-non-pad-token (EOS)
# pooling + L2 normalize, document mode (no instruction), use_cache=False.
# gte-Qwen2 is BIDIRECTIONAL (config.is_causal=False); its remote modeling_qwen.py
# is the only faithful encoder (native Qwen2 on tf-5.5 is causal-only -> wrong,
# cos~0 vs official). The remote code runs on transformers 5.5 via the rope_theta
# shim + use_cache=False (no flash-attn / KV cache needed for a single forward).
_AUTOMODEL = {
    "f2llm":    {"trust_remote_code": False, "attn_implementation": None,    "rope_theta_patch": False},
    "gteqwen2": {"trust_remote_code": True,  "attn_implementation": "eager", "rope_theta_patch": True,
                 "needs_tf44": True},
}


def is_automodel(llm: str) -> bool:
    return llm in _AUTOMODEL


def _patch_qwen2_rope_theta():
    """gte-Qwen2's 4.39-era remote code reads config.rope_theta, which transformers
    5.x dropped from Qwen2Config. Expose it (1e6 for Qwen2-7B). Only used inside
    the tf-4.44 subprocess fallback / native tf-4.x; a no-op when already present."""
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    if "rope_theta" not in Qwen2Config.__dict__:
        Qwen2Config.rope_theta = property(lambda self: 1000000.0)


def _encode_gteqwen2_via_tf44(texts, batch_size: int) -> np.ndarray:
    """gte-Qwen2 is BIDIRECTIONAL and only computes correctly under its native
    transformers (~4.44). On transformers 5.x we encode via a subprocess into the
    side-installed tf-4.44 env (+ flash_attn stub) so the official remote code runs
    faithfully (verified vs the model card). See clear3d/_gteqwen2_worker.py."""
    import os, sys, json, subprocess, tempfile
    repo_root = Path(__file__).resolve().parents[1]
    compat = os.environ.get("CLIP3D_GTEQWEN2_TF44", str(repo_root / ".tf44_gteqwen2"))
    worker = str(Path(__file__).resolve().parent / "_gteqwen2_worker.py")
    if not (Path(compat) / "transformers").exists():
        raise RuntimeError(
            f"gte-Qwen2 needs the tf-4.44 side-env at {compat} (not found). Build it:\n"
            f"  pip install --no-deps --target={compat} transformers==4.44.2 "
            f"tokenizers==0.19.1 huggingface_hub==0.24.6\n"
            f"  mkdir -p {compat}/flash_attn && touch {compat}/flash_attn/__init__.py")
    with tempfile.TemporaryDirectory() as td:
        ij = os.path.join(td, "in.json"); on = os.path.join(td, "out.npy")
        with open(ij, "w") as f:
            json.dump(list(texts), f)
        env = dict(os.environ)
        env["PYTHONPATH"] = compat + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, worker, ij, on, str(batch_size)],
                       env=env, check=True)
        return np.load(on).astype(np.float32)


@torch.no_grad()
def encode_with_automodel(model_name: str, texts, dtype, *, batch_size: int = 16,
                          trust_remote_code: bool = False,
                          attn_implementation=None, rope_theta_patch: bool = False,
                          max_length: int = 8192, show_progress: bool = False,
                          needs_tf44: bool = False) -> np.ndarray:
    """Official model-card encoder: AutoModel + last-non-pad-token pooling +
    L2 normalize, document mode (no instruction prefix), use_cache=False.
    `needs_tf44` (gte-Qwen2): on transformers 5.x its bidirectional remote code is
    wrong, so route through the tf-4.44 subprocess instead."""
    import transformers
    if needs_tf44 and int(transformers.__version__.split(".")[0]) >= 5:
        return _encode_gteqwen2_via_tf44(texts, batch_size)
    from transformers import AutoModel, AutoTokenizer
    if rope_theta_patch:
        _patch_qwen2_rope_theta()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    mk = {"dtype": dtype, "low_cpu_mem_usage": True, "trust_remote_code": trust_remote_code}
    if attn_implementation:
        mk["attn_implementation"] = attn_implementation
    model = AutoModel.from_pretrained(model_name, **mk).to("cuda").eval()
    rng = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm
        rng = tqdm(rng, desc=model_name)
    out = []
    for i in rng:
        chunk = texts[i:i + batch_size]
        bd = tok(chunk, max_length=max_length, padding=True, truncation=True,
                 return_tensors="pt").to("cuda")
        lhs = model(**bd, use_cache=False).last_hidden_state
        am = bd["attention_mask"]
        if int(am[:, -1].sum()) == am.shape[0]:        # left-padded
            emb = lhs[:, -1]
        else:                                          # right-padded -> last non-pad (EOS)
            sl = am.sum(dim=1) - 1
            emb = lhs[torch.arange(lhs.shape[0], device=lhs.device), sl]
        emb = F.normalize(emb, p=2, dim=1)
        out.append(emb.float().cpu().numpy())
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(out, 0).astype(np.float32)


def encode_label_texts(llm: str, texts) -> np.ndarray:
    """Encode `texts` with backend `llm`, L2-normalized. openai -> API,
    f2llm/gteqwen2 -> official AutoModel, sfr/harrier -> SentenceTransformer."""
    if llm == "openai":
        return _encode_with_openai(_LLM_REPO[llm], texts)
    if is_automodel(llm):
        return encode_with_automodel(_LLM_REPO[llm], texts, _LLM_DTYPE.get(llm, "auto"),
                                     **_AUTOMODEL[llm])
    _, dtype, mkw, doc, trc = llm_st_cfg(llm)
    return _encode_with_st(_LLM_REPO[llm], texts, dtype=dtype, model_kwargs=mkw,
                           document_mode=doc, trust_remote_code=trc)


def _encode_with_st(model_name: str, texts: list[str], dtype,
                    model_kwargs: dict | None = None,
                    document_mode: bool = False,
                    batch_size: int = 8,
                    trust_remote_code: bool = True) -> np.ndarray:
    """SentenceTransformer encode (matches get_embed_ct.py).

    `model_kwargs` merges into the from_pretrained kwargs (e.g.
    attn_implementation="sdpa"). When `document_mode`, encode via
    `encode_document` so instruction-tuned retrieval encoders don't prepend a
    query prompt — keeping labels symmetric with the document-encoded bank.
    `trust_remote_code=False` falls back to the model's native transformers
    architecture (needed for gte-Qwen2 under transformers 5.5).
    """
    from sentence_transformers import SentenceTransformer
    mk = {"dtype": dtype, "low_cpu_mem_usage": True}
    if model_kwargs:
        mk.update(model_kwargs)
    model = SentenceTransformer(
        model_name,
        model_kwargs=mk,
        trust_remote_code=trust_remote_code,
    ).to("cuda")
    enc = model.encode_document if document_mode else model.encode
    emb = enc(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return emb


class OpenAIUnavailable(RuntimeError):
    """Raised when the OpenAI embeddings API can't be reached and no cache
    exists. Catching code should skip the openai method gracefully and tell
    the user to run precompute_openai_labels.py on a network-accessible host
    and copy the resulting label_emb.openai*.npz into outputs/<tag>/cache/."""


def _encode_with_openai(model_name: str, texts: list[str],
                        dimensions: int | None = None,
                        batch_size: int = 256) -> np.ndarray:
    """OpenAI embeddings API. Requires `OPENAI_API_KEY`. Output L2-normalized.

    On no-internet machines (e.g. our GH200 nodes that block api.openai.com)
    this raises OpenAIUnavailable; callers are expected to catch and skip.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
    except Exception as e:
        raise OpenAIUnavailable(f"openai import/client init failed: {e}") from e

    cleaned = [t.replace("\\n", " ").replace("\n", " ").strip() for t in texts]
    extra = {"dimensions": dimensions} if dimensions else {}
    out = []
    try:
        for i in range(0, len(cleaned), batch_size):
            chunk = cleaned[i:i + batch_size]
            resp = client.embeddings.create(model=model_name, input=chunk,
                                            encoding_format="float", **extra)
            out.extend(d.embedding for d in resp.data)
    except Exception as e:
        raise OpenAIUnavailable(
            f"OpenAI API call failed ({type(e).__name__}: {e}). "
            f"Run precompute_openai_labels.py on a host with internet, then "
            f"copy outputs/<tag>/cache/label_emb.openai*.npz onto this machine."
        ) from e
    emb = np.asarray(out, dtype=np.float32)
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return emb / n


def get_label_embeddings(llm: str, *, overwrite: bool = False) -> dict:
    """Returns {pos: (n_labels, D), neg: (n_labels, D), labels: [...]}."""
    out = cache_label_emb(llm)
    if out.exists() and not overwrite:
        z = np.load(out, allow_pickle=True)
        return {"pos": z["pos"], "neg": z["neg"], "labels": list(z["labels"])}

    if llm not in _LLM_REPO:
        raise ValueError(f"Unknown llm '{llm}', expected one of {list(_LLM_REPO)}")
    model_name = _LLM_REPO[llm]

    pos_texts = [lbl.lower() for lbl in LABELS_18]
    neg_texts = [f"no {lbl.lower()}" for lbl in LABELS_18]

    print(f"[llm_prompts:{llm}] encoding {2*len(LABELS_18)} prompts via {model_name}")
    all_emb = encode_label_texts(llm, pos_texts + neg_texts)
    pos = all_emb[: len(LABELS_18)]
    neg = all_emb[len(LABELS_18):]

    np.savez(out, pos=pos, neg=neg, labels=np.asarray(LABELS_18, dtype=object))
    print(f"[llm_prompts:{llm}] saved → {out}  pos={pos.shape} neg={neg.shape}")
    return {"pos": pos, "neg": neg, "labels": LABELS_18}

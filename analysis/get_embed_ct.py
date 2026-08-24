#!/usr/bin/env python3
"""
Step 2 of the CLEAR-3D pipeline.

1. Reads CT-RATE + MERLIN concept JSONLs.
2. Applies the *exact* CLEAR cleaning recipe: lowercase + python `set()` dedup.
3. Embeds the unique-concept list into four spaces:
   - Salesforce/SFR-Embedding-Mistral  (4096-d)   — CLEAR's published recipe
   - microsoft/harrier-oss-v1-27b      (5376-d)   — current SOTA general embedder
   - OpenAI text-embedding-3-large     (3072-d)   — managed-API embedder
   - 3D-CLIP text tower (default model 768-d)     — needed for img@concept routing
4. Saves /path/to/ACT/analysis/concept_bank.pkl.
"""
import argparse, ast, gc, json, os, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from clear3d.llm_prompts import encode_with_automodel, _AUTOMODEL

# All paths are env-overridable so this script runs on any machine. Defaults
# match our cluster layout. SFR / Harrier embedding has no 3D-CLIP dependency
# and runs anywhere with `pip install sentence-transformers torch`.
ROOT = Path(os.environ.get("CLIP3D_CONCEPTS_ROOT",
                           str(Path(__file__).resolve().parent)))
JSONL_FILES = os.environ.get("CONCEPT_JSONL_FILES",
                             f"{ROOT}/ctrate_concepts.jsonl,"
                             f"{ROOT}/merlin_concepts.jsonl").split(",")
OUT_PATH = os.environ.get("CONCEPT_BANK_PKL", str(ROOT / "concept_bank.pkl"))

CONFIGS_JSON = os.environ.get("CLIP3D_CONFIGS",
                              "/path/to/ACT/analysis/eval/configs.json")
CLIP_REPO = os.environ.get("CLIP3D_REPO",
                           "/path/to/ACT/model")
EVAL_REPO = os.environ.get("CLIP3D_EVAL_REPO",
                           "/path/to/ACT/analysis/eval")
CLIP3D_BEST = os.environ.get(
    "CLIP3D_BEST", "clip_3d_ctrate_merlin_dinov3_vitb_transformers_allgather"
)


def parse_obs(rec):
    raw = rec.get("model_output", "")
    if not raw:
        return []
    try:
        d = json.loads(raw)
    except Exception:
        try:
            d = ast.literal_eval(raw)
        except Exception:
            return []
    obs = d.get("observations") or []
    return [o for o in obs if isinstance(o, str)]


def build_concept_list():
    seen = set()
    for f in JSONL_FILES:
        with open(f) as fh:
            for line in fh:
                rec = json.loads(line)
                for o in parse_obs(rec):
                    seen.add(o.lower().strip())
    seen.discard("")
    concepts = sorted(seen)  # deterministic order
    return concepts


# Per CLEAR (examples/concepts/get_embed.py) SFR-Embedding-Mistral is an
# instruction-tuned encoder and is wrapped as `Instruct: {task}\nQuery: {text}`
# before encoding. Without this prefix the embedding quality degrades versus
# CLEAR's published recipe. We do NOT prefix Harrier (out-of-CLEAR addition).
SFR_TASK = ("Given medical radiology concepts, retrieve relevant "
            "passages and generate embeddings")


def _maybe_wrap_instruct(model_name, concepts):
    if "SFR-Embedding-Mistral" in model_name:
        return [f"Instruct: {SFR_TASK}\nQuery: {c}" for c in concepts]
    return concepts


def encode_with_sentence_transformer(model_name, concepts, batch_size, dtype,
                                     model_kwargs=None, document_mode=False,
                                     trust_remote_code=True):
    from sentence_transformers import SentenceTransformer

    print(f"\n[{model_name}] loading...")
    t0 = time.time()
    mk = {"dtype": dtype, "low_cpu_mem_usage": True}
    if model_kwargs:
        mk.update(model_kwargs)
    model = SentenceTransformer(
        model_name,
        model_kwargs=mk,
        trust_remote_code=trust_remote_code,
    )
    model = model.to("cuda")
    print(f"[{model_name}] loaded in {time.time()-t0:.0f}s; embedding {len(concepts)} concepts...")

    inputs = _maybe_wrap_instruct(model_name, concepts)
    # document_mode -> encode_document (no query instruction), for instruction-
    # tuned retrieval encoders (f2llm, gteqwen2); keeps concepts symmetric with
    # the document-encoded labels in clear3d/llm_prompts.py.
    enc = model.encode_document if document_mode else model.encode
    t0 = time.time()
    emb = enc(
        inputs,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"[{model_name}] {emb.shape}  in {time.time()-t0:.0f}s")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return emb.astype(np.float32)


def encode_with_openai(model_name, concepts, batch_size, dimensions=None,
                       max_retries=6):
    """Embed `concepts` via the OpenAI embeddings API and L2-normalize.

    Requires OPENAI_API_KEY in the environment. `text-embedding-3-large` is
    3072-d by default; pass `dimensions` to shorten (Matryoshka)."""
    from openai import OpenAI

    client = OpenAI()
    print(f"\n[{model_name}] embedding {len(concepts)} concepts via OpenAI API...")
    t0 = time.time()
    out = []
    # Mirror CLEAR's Azure path: strip newlines before send.
    cleaned = [c.replace("\\n", " ").replace("\n", " ").strip() for c in concepts]
    extra = {"dimensions": dimensions} if dimensions else {}
    for i in tqdm(range(0, len(cleaned), batch_size), desc=model_name):
        chunk = cleaned[i : i + batch_size]
        delay = 1.0
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(
                    model=model_name,
                    input=chunk,
                    encoding_format="float",
                    **extra,
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"[{model_name}] batch {i} attempt {attempt+1} failed: {e}; retrying in {delay:.1f}s")
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        # API preserves input order in resp.data
        out.extend(d.embedding for d in resp.data)
    emb = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    print(f"[{model_name}] {emb.shape}  in {time.time()-t0:.0f}s")
    return emb


def encode_with_clip_text_tower(concepts, batch_size=128, context_length=77):
    """Use our 3D-CLIP text encoder."""
    sys.path.insert(0, CLIP_REPO)
    sys.path.insert(0, EVAL_REPO)
    import clip  # noqa: E402
    from clip import _tokenizer  # noqa: E402
    from eval_all import build_and_load  # noqa: E402

    cfgs = json.load(open(CONFIGS_JSON))
    cfg = cfgs[CLIP3D_BEST]
    print(f"\n[3D-CLIP text tower] loading {cfg['model_dir']}...")
    t0 = time.time()
    model = build_and_load(cfg, context_length=context_length)
    model = model.to("cuda").eval()
    print(f"[3D-CLIP] loaded in {time.time()-t0:.0f}s; embedding {len(concepts)} concepts...")

    # Pre-truncate over-long concepts at the BPE-token level so clip.tokenize never raises.
    # Reserve 2 slots for <|startoftext|> and <|endoftext|>.
    max_body = context_length - 2
    safe_concepts = []
    n_truncated = 0
    for c in concepts:
        toks = _tokenizer.encode(c)
        if len(toks) > max_body:
            c = _tokenizer.decode(toks[:max_body])
            n_truncated += 1
        safe_concepts.append(c)
    print(f"[3D-CLIP] truncated {n_truncated}/{len(concepts)} concepts to fit context_length={context_length}")

    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(safe_concepts), batch_size), desc="clip_text"):
            chunk = safe_concepts[i : i + batch_size]
            tok = clip.tokenize(chunk, context_length).to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                f = model.encode_text(tok)
            f = f.float()
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().numpy())
    emb = np.concatenate(out, axis=0).astype(np.float32)
    print(f"[3D-CLIP] {emb.shape}  in {time.time()-t0:.0f}s")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return emb


def _partial_path(out_path, key):
    p = Path(out_path)
    # The clip_text encoding depends on which 3D-CLIP checkpoint is used.
    # When CLIP3D_RUN_TAG is set, suffix the file so different image encoders
    # don't overwrite each other's text-tower bank.
    tag = os.environ.get("CLIP3D_RUN_TAG", "")
    if key == "clip_text_emb" and tag:
        return p.parent / f"{p.stem}.{key}.{tag}.npz"
    return p.parent / f"{p.stem}.{key}.npz"


def save_partial(out_path, key, emb, concepts):
    path = _partial_path(out_path, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # Pass an open file handle so np.savez does NOT auto-append ".npz" to the name.
    with open(tmp, "wb") as fh:
        np.savez(fh, emb=emb, concepts=np.asarray(concepts, dtype=object))
    os.replace(tmp, path)
    print(f"[ckpt] saved {key} → {path}  shape={emb.shape}")


def load_partial(out_path, key, expected_concepts):
    path = _partial_path(out_path, key)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    cached = list(z["concepts"])
    if cached != expected_concepts:
        print(f"[ckpt] {path} concept list mismatch ({len(cached)} vs {len(expected_concepts)}); ignoring")
        return None
    emb = z["emb"].astype(np.float32, copy=False)
    print(f"[ckpt] resumed {key} ← {path}  shape={emb.shape}")
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--skip_sfr", action="store_true")
    ap.add_argument("--skip_harrier", action="store_true")
    ap.add_argument("--skip_openai", action="store_true")
    ap.add_argument("--skip_clip", action="store_true")
    ap.add_argument("--skip_f2llm", action="store_true")
    ap.add_argument("--skip_gteqwen2", action="store_true")
    ap.add_argument("--sfr_bs", type=int, default=32)
    ap.add_argument("--harrier_bs", type=int, default=8)
    ap.add_argument("--openai_bs", type=int, default=256)
    ap.add_argument("--openai_model", default="text-embedding-3-large")
    ap.add_argument("--openai_dims", type=int, default=None,
                    help="Optional Matryoshka shortening (e.g. 1024).")
    ap.add_argument("--clip_bs", type=int, default=128)
    ap.add_argument("--f2llm_bs", type=int, default=64)
    ap.add_argument("--gteqwen2_bs", type=int, default=128)
    ap.add_argument("--skip_pickle", action="store_true",
                    help="Do not rewrite the combined concept_bank.pkl (use for "
                         "partial builds so existing banks aren't clobbered).")
    args = ap.parse_args()

    concepts = build_concept_list()
    print(f"unique concepts: {len(concepts)}")
    print("first 5:", concepts[:5])

    bank = {"concepts": concepts}

    # 1) SFR-Mistral (CLEAR baseline)
    if not args.skip_sfr:
        emb = load_partial(args.out, "sfr_emb", concepts)
        if emb is None:
            emb = encode_with_sentence_transformer(
                "Salesforce/SFR-Embedding-Mistral",
                concepts,
                batch_size=args.sfr_bs,
                dtype=torch.bfloat16,
            )
            save_partial(args.out, "sfr_emb", emb, concepts)
        bank["sfr_emb"] = emb

    # 2) Harrier-OSS-27B (current SOTA)
    if not args.skip_harrier:
        emb = load_partial(args.out, "harrier_emb", concepts)
        if emb is None:
            emb = encode_with_sentence_transformer(
                "microsoft/harrier-oss-v1-27b",
                concepts,
                batch_size=args.harrier_bs,
                dtype="auto",
            )
            save_partial(args.out, "harrier_emb", emb, concepts)
        bank["harrier_emb"] = emb

    # 3) OpenAI text-embedding-3-large
    if not args.skip_openai:
        emb = load_partial(args.out, "openai_emb", concepts)
        if emb is None:
            emb = encode_with_openai(
                args.openai_model,
                concepts,
                batch_size=args.openai_bs,
                dimensions=args.openai_dims,
            )
            save_partial(args.out, "openai_emb", emb, concepts)
        bank["openai_emb"] = emb

    # 4) 3D-CLIP text tower
    if not args.skip_clip:
        emb = load_partial(args.out, "clip_text_emb", concepts)
        if emb is None:
            emb = encode_with_clip_text_tower(
                concepts, batch_size=args.clip_bs
            )
            save_partial(args.out, "clip_text_emb", emb, concepts)
        bank["clip_text_emb"] = emb

    # 5) F2LLM-v2-14B  — official AutoModel + EOS pooling (document mode); 5120-d
    if not args.skip_f2llm:
        emb = load_partial(args.out, "f2llm_emb", concepts)
        if emb is None:
            emb = encode_with_automodel(
                "codefuse-ai/F2LLM-v2-14B", concepts, torch.bfloat16,
                batch_size=args.f2llm_bs, show_progress=True, **_AUTOMODEL["f2llm"])
            save_partial(args.out, "f2llm_emb", emb, concepts)
        bank["f2llm_emb"] = emb

    # 6) gte-Qwen2-7B-instruct — official BIDIRECTIONAL remote code (last-token
    #    pooling, document mode); 3584-d. gte-Qwen2 sets config.is_causal=False;
    #    native Qwen2 on tf-5.5 is causal-only (wrong, cos~0 vs official), so we
    #    run the remote modeling via the rope_theta shim + use_cache=False — see
    #    clear3d.llm_prompts.encode_with_automodel.
    if not args.skip_gteqwen2:
        emb = load_partial(args.out, "gteqwen2_emb", concepts)
        if emb is None:
            emb = encode_with_automodel(
                "Alibaba-NLP/gte-Qwen2-7B-instruct", concepts, torch.bfloat16,
                batch_size=args.gteqwen2_bs, show_progress=True, **_AUTOMODEL["gteqwen2"])
            save_partial(args.out, "gteqwen2_emb", emb, concepts)
        bank["gteqwen2_emb"] = emb

    if args.skip_pickle:
        print(f"\n--skip_pickle: leaving {args.out} untouched "
              "(per-backend npz partials are saved).")
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(bank, f, protocol=pickle.HIGHEST_PROTOCOL)
    sizes = {k: (v.shape if hasattr(v, "shape") else len(v)) for k, v in bank.items()}
    print(f"\nwrote {args.out}")
    print("schema:", sizes)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Embed the 376,194 concept strings with BiomedBERT (mean-pooled last hidden state)
as a weak text-encoder baseline for the concept-space comparison. Saves
concept_bank.biomedbert_emb.npz with members {emb: (N,768) f32, concepts}.

Untuned masked-LM embeddings (mean-pooled) are a deliberately modest baseline;
the point is to show the dedicated LLM embedder (f2llm) yields a far more
semantically coherent concept space than a domain BERT or the CLIP text tower.
"""
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parent
MODELS = [
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
]
BATCH, MAXLEN = 256, 64

def load_model():
    last = None
    for name in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name)
            print(f"loaded {name}", flush=True)
            return tok, mdl, name
        except Exception as e:
            last = e; print(f"  {name} failed: {e}", flush=True)
    raise last

def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    concepts = np.load(ROOT / "concept_bank.f2llm_emb.npz", allow_pickle=True)["concepts"]
    texts = [str(c) for c in concepts]
    n = len(texts)
    tok, mdl, name = load_model()
    mdl.eval().to(dev)
    out = np.empty((n, mdl.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, BATCH):
            batch = texts[s:s + BATCH]
            enc = tok(batch, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
            hs = mdl(**enc).last_hidden_state                       # (b, L, H)
            m = enc["attention_mask"].unsqueeze(-1).float()          # (b, L, 1)
            emb = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)         # masked mean pool
            out[s:s + len(batch)] = emb.float().cpu().numpy()
            if (s // BATCH) % 100 == 0:
                el = time.time() - t0
                print(f"  {s + len(batch):>7}/{n}  [{el:.0f}s, {(s+len(batch))/max(el,1):.0f}/s]", flush=True)
    np.savez(ROOT / "concept_bank.biomedbert_emb.npz", emb=out, concepts=concepts)
    print(f"saved concept_bank.biomedbert_emb.npz  emb={out.shape}  model={name}  [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    main()

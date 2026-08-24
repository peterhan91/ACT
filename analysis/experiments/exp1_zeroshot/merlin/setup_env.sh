#!/bin/bash
# aarch64 env for Merlin (Stanford 3D abdominal-CT VLM) on the GH200.
#
# merlin-vlm is pure-Python; the only native deps (torch/torchvision/sentencepiece/
# tokenizers/safetensors) all ship cu130 aarch64 wheels — the SAME stack already
# proven working in the `crimson` env on this node. We reuse the node's known-good
# torch 2.11.0+cu130 and install merlin-vlm with --no-deps so it can't pull a
# CPU/x86 torch from plain PyPI. No flash-attn / apex / mmcv anywhere.
set -euo pipefail
ENV=merlin
PYVER=3.12
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
MAMBA=${MAMBA_EXE:-/path/to/micromamba}
PREFIX=/path/to/conda/envs/$ENV
ENVPY=$PREFIX/bin/python

echo "=== [1/4] create env $ENV (python $PYVER) at $PREFIX ==="
"$MAMBA" create -y -p "$PREFIX" -c conda-forge "python=$PYVER" pip

echo "=== [2/4] torch/torchvision (cu130 aarch64, matches crimson) ==="
"$ENVPY" -m pip install --index-url "$TORCH_INDEX" torch==2.11.0 torchvision==0.26.0

echo "=== [3/4] merlin-vlm (--no-deps so it can't override torch) + light deps ==="
"$ENVPY" -m pip install --no-deps merlin-vlm
"$ENVPY" -m pip install "monai>=1.3.0" "transformers>=4.38.2" huggingface_hub \
  nibabel nltk pandas rich einops peft matplotlib sentencepiece protobuf accelerate "numpy>=1.26.4"

echo "=== [4/4] smoke test (import only — does NOT download the 1GB checkpoint) ==="
"$ENVPY" - <<'PY'
import torch
from merlin import Merlin            # class import; weights download on first Merlin() call
print("merlin import OK | torch", torch.__version__, "| cuda", torch.cuda.is_available())
PY

cat <<'NOTE'
Done. Weights (stanfordmimi/Merlin, PUBLIC/ungated) auto-download on the first
Merlin() instantiation. Pre-cache on a login node with internet:
  export HF_HOME=/path/to/.cache/huggingface
  python -c "from merlin import Merlin; Merlin()"   # pulls ~1GB ckpt + Clinical-Longformer + ResNet152
Zero-shot is image<->report contrastive similarity (Merlin() -> img_emb[512], txt_emb[512])
or the built-in 1692-phecode head (Merlin(PhenotypeCls=True)). See README.md.
NOTE

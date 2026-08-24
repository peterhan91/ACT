#!/bin/bash
# aarch64 env for CT-CLIP (3D chest-CT CLIP; CT-RATE authors). No custom CUDA
# kernels, so ARM-clean ONCE we ignore the repo's commented-out torch==2.0.1
# (no aarch64 wheel) and avoid the old transformers==4.30.1 -> tokenizers-without-
# aarch64-wheel trap by installing a modern transformers.
set -euo pipefail
ENV=ctclip
PYVER=3.11
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
MAMBA=${MAMBA_EXE:-/path/to/micromamba}
PREFIX=/path/to/conda/envs/$ENV
ENVPY=$PREFIX/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== [1/5] create env $ENV (python $PYVER) at $PREFIX ==="
"$MAMBA" create -y -p "$PREFIX" -c conda-forge "python=$PYVER" pip

echo "=== [2/5] recent aarch64 CUDA torch (NOT the repo's 2.0.1) ==="
"$ENVPY" -m pip install --index-url "$TORCH_INDEX" torch torchvision

echo "=== [3/5] clone CT-CLIP ==="
[ -d "$HERE/CT-CLIP" ] || git clone https://github.com/ibrahimethemhamamci/CT-CLIP.git "$HERE/CT-CLIP"

echo "=== [4/5] core deps (modern transformers; keep vector-quantize-pytorch==1.1.2) ==="
"$ENVPY" -m pip install "transformers>=4.40,<4.50" accelerate huggingface_hub sentencepiece \
  "vector-quantize-pytorch==1.1.2" einops ema-pytorch beartype torchtyping ftfy regex \
  nibabel opencv-python pillow nltk h5py XlsxWriter openpyxl pandas tqdm matplotlib seaborn

echo "=== [5/5] editable installs of the two helper pkgs, torch-safe (--no-deps) ==="
"$ENVPY" -m pip install -e "$HERE/CT-CLIP/transformer_maskgit" --no-deps
"$ENVPY" -m pip install -e "$HERE/CT-CLIP/CT_CLIP" --no-deps

"$ENVPY" - <<'PY'
import torch
from transformer_maskgit import CTViT
from ct_clip import CTCLIP
print("ct-clip import OK | torch", torch.__version__, "| cuda", torch.cuda.is_available())
PY

cat <<NOTE
Done (env). WEIGHTS ARE GATED:
  1) accept ToS at https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
  2) "$ENVPY" -m huggingface_hub.commands.huggingface_cli login   (or: huggingface-cli login)
  3) "$PREFIX/bin/huggingface-cli" download ibrahimhamamci/CT-RATE \\
       models/CT-CLIP-Related/CT-CLIP_v2.pt --repo-type dataset --local-dir "$HERE/weights"
PATCH for torch>=2.6: in CT-CLIP/CT_CLIP/ct_clip/ct_clip.py load():
  torch.load(str(path), map_location="cpu", weights_only=False)
NOTE

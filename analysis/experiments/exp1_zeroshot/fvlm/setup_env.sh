#!/bin/bash
# aarch64 env for f-VLM (Alibaba DAMO, ICLR'25 3D-CT VLM; PyTorch + vendored
# LAVIS/BLIP). NO JAX/mmcv/detectron2. Only blocker is the repo's torch==2.3.1
# (no aarch64 wheel) -> bump to 2.7.1+cu128 (aarch64 wheel exists; runs on the
# CUDA-13 driver). Installs the MINIMAL zero-shot-eval deps only (skips
# nnunet/TotalSegmentator/spacy/RaTEScore, which are mask-gen / report metrics).
set -euo pipefail
ENV=fvlm
PYVER=3.10
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
MAMBA=${MAMBA_EXE:-/path/to/micromamba}
PREFIX=/path/to/conda/envs/$ENV
ENVPY=$PREFIX/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== [1/4] create env $ENV (python $PYVER) at $PREFIX ==="
"$MAMBA" create -y -p "$PREFIX" -c conda-forge "python=$PYVER" pip

echo "=== [2/4] torch 2.7.1+cu128 (aarch64; replaces repo's x86-only 2.3.1) ==="
"$ENVPY" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "$TORCH_INDEX"

echo "=== [3/4] clone fvlm ==="
[ -d "$HERE/fvlm" ] || git clone https://github.com/alibaba-damo-academy/fvlm.git "$HERE/fvlm"

echo "=== [4/4] minimal eval deps (transformers MUST stay <4.27 for vendored BLIP) ==="
"$ENVPY" -m pip install monai==1.3.2 nibabel==5.2.1 SimpleITK==2.4.0 \
  "connected-components-3d>=3.19.0" scikit-image==0.24.0 einops==0.8.0 timm==0.4.12 \
  fairscale==0.4.13 opencv-python-headless==4.5.5.64 pandas==2.2.2 "numpy<2" \
  scikit-learn==1.5.1 tqdm pyyaml omegaconf \
  transformers==4.25.0 tokenizers==0.13.3 huggingface_hub==0.25.0 sentencepiece==0.2.0 gdown

"$ENVPY" - <<'PY'
import torch, monai, transformers
print("fvlm deps OK | torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__)
PY

cat <<NOTE
Done (env). WEIGHTS (Google Drive, ungated):
  "$ENVPY" -m gdown --folder 15BnMo1lIAlOH_8KLdB2NugiHnmj9AWSD -O "$HERE/ckpts"
NOTE: f-VLM eval also needs TotalSegmentator organ masks + (112,256,352) NIfTI
preprocessing — build masks in a SEPARATE env (TotalSegmentator pulls nnunetv2).
Text encoder microsoft/BiomedVLP-CXR-BERT-specialized auto-downloads at load.
NOTE

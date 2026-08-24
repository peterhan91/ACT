#!/bin/bash
# Build a TotalSegmentator env WITHOUT conda (the system anaconda is x86, can't run on
# this aarch64 node). Bootstrap a venv from crimson's aarch64 python (3.12), then pip
# install an aarch64 CUDA torch (cu130, matches crimson) + TotalSegmentator. TS model
# weights download from Zenodo/GitHub on first inference (both reachable).
set -e
BASE=python
VENV=/path/to/conda/envs/totalseg_venv
echo "=== [1/3] venv from aarch64 crimson python ==="
"$BASE" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip -q
echo "=== [2/3] torch (aarch64 cu130) ==="
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cu130 torch
echo "=== [3/3] TotalSegmentator ==="
"$VENV/bin/pip" install TotalSegmentator nibabel
"$VENV/bin/python" -c "import torch, totalsegmentator; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('TS import OK')"
echo "SETUP DONE"

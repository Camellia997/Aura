#!/bin/bash
# =====================================================================
# Environment setup for HunyuanVideo edit (opensource inference).
#
# Prerequisites:
#   - conda (miniconda / anaconda) available on PATH
#   - an NVIDIA driver supporting CUDA 12.4
#
# Usage:
#   bash jobs/install.sh                       # creates env "aura"
#   ENV_NAME=myenv PYTHON_VERSION=3.10 bash jobs/install.sh
# =====================================================================
set -e

ENV_NAME="${ENV_NAME:-aura}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

# (Optional) proxy — uncomment and edit if your machine needs one to reach pypi.
# export http_proxy="http://<proxy_host>:<port>"
# export https_proxy="http://<proxy_host>:<port>"

# ---------------------------------------------------------------------
# 1) Create and activate a fresh conda environment
# ---------------------------------------------------------------------
eval "$(conda shell.bash hook)"
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip

# ---------------------------------------------------------------------
# 2) PyTorch stack (CUDA 12.4 wheels)
# ---------------------------------------------------------------------
pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124
# Pin cuBLAS to the version validated with these wheels.
pip3 install nvidia-cublas-cu12==12.4.5.8

# ---------------------------------------------------------------------
# 3) transformers + diffusers (versions verified against this codebase)
# ---------------------------------------------------------------------
pip3 install transformers==4.57.1
pip3 install "diffusers @ git+https://github.com/huggingface/diffusers@71a6fd9f0df04d3764dfa999268a05d87903a85a"

# ---------------------------------------------------------------------
# 4) Remaining python dependencies
# ---------------------------------------------------------------------
pip3 install -r "$PROJ_DIR/requirements.txt"

# ---------------------------------------------------------------------
# 5) flash-attention (built against the installed torch; no build isolation)
# ---------------------------------------------------------------------
pip3 install flash_attn==2.7.4.post1 --no-build-isolation

echo ""
echo "==============================================================="
echo " Environment '$ENV_NAME' is ready."
echo " Activate it with:  conda activate $ENV_NAME"
echo "==============================================================="

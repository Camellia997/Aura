#!/bin/bash
# =====================================================================
# Download model weights for Aura inference (wraps download.py):
#   - Wan-AI/Wan2.2-T2V-A14B        -> base checkpoints (--ckpt_dir)
#   - Qwen/Qwen2.5-VL-3B-Instruct   -> meta-query VLM backbone (--vlm_dir)
#   - Camellia997/Aura              -> finetuned Aura high/low-noise experts
#                                      (--high_noise_model_dir / --low_noise_model_dir)
#
# Usage:
#   bash jobs/download.sh                                  # all three -> ./weights
#   OUTPUT_DIR=/data/weights bash jobs/download.sh         # custom directory
#   MODELS=wan  bash jobs/download.sh                      # only Wan2.2
#   MODELS=qwen bash jobs/download.sh                      # only Qwen2.5-VL
#   MODELS=aura bash jobs/download.sh                      # only Aura experts
#   HF_ENDPOINT=https://hf-mirror.com bash jobs/download.sh  # use a mirror
#   HF_TOKEN=hf_xxx bash jobs/download.sh                  # gated/private repos
# =====================================================================
set -e

# Run from the project root (the directory that contains download.py)
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-weights}"
MODELS="${MODELS:-all}"             # all | both | wan | qwen | aura
MAX_WORKERS="${MAX_WORKERS:-8}"
ENABLE_HF_TRANSFER="${ENABLE_HF_TRANSFER:-0}"   # set to 1 to use the hf_transfer backend

EXTRA_ARGS=()
if [ "$ENABLE_HF_TRANSFER" = "1" ]; then
    EXTRA_ARGS+=("--enable_hf_transfer")
fi
if [ -n "$HF_TOKEN" ]; then
    EXTRA_ARGS+=("--hf_token" "$HF_TOKEN")
fi
if [ -n "$HF_ENDPOINT" ]; then
    EXTRA_ARGS+=("--endpoint" "$HF_ENDPOINT")
fi

python download.py \
    --output_dir "$OUTPUT_DIR" \
    --models "$MODELS" \
    --max_workers "$MAX_WORKERS" \
    "${EXTRA_ARGS[@]}"

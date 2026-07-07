#!/bin/bash
# =====================================================================
# Single-GPU inference for HunyuanVideo edit (T5 + metaquery, high/low DiT)
# =====================================================================
set -e

# Run from the project root (the directory that contains infer_hyvideo_edit.py)
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------
# Paths (fill in before running)
# ---------------------------------------------------------------------
CKPT_DIR="<PATH_TO_BASE_CKPT_DIR>"                 # contains T5 + VAE (e.g. Wan2.2-T2V-A14B)
HIGH_NOISE_MODEL_DIR="<PATH_TO_HIGH_NOISE_DIR>"    # directory with high-noise DiT .bin weights
LOW_NOISE_MODEL_DIR="<PATH_TO_LOW_NOISE_DIR>"      # directory with low-noise DiT .bin weights
VLM_DIR="Qwen/Qwen2.5-VL-3B-Instruct"             # Qwen2.5-VL model path or HF id
CSV_FILE="<PATH_TO_VALIDATION_JSON>"               # validation set (json)
SAVE_DIR="results/single_gpu"

# ---------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------
RESOLUTION="720*1280"   # 720P vertical: 720*1280, horizontal: 1280*720
GEN_MODE="ref"
BOUNDARY=0.875

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python inference.py \
    --size $RESOLUTION \
    --ckpt_dir "$CKPT_DIR" \
    --high_noise_model_dir "$HIGH_NOISE_MODEL_DIR" \
    --low_noise_model_dir "$LOW_NOISE_MODEL_DIR" \
    --vlm_dir "$VLM_DIR" \
    --csv_file "$CSV_FILE" \
    --save_dir "$SAVE_DIR" \
    --log_file "$SAVE_DIR/logging.txt" \
    --gen_mode $GEN_MODE \
    --boundary $BOUNDARY \
    --base_seed 1024 \
    --sample_shift 6.0 \
    --offload_model True \
    --convert_model_dtype \
    --apply_rope_in_selfattn False \
    --ulysses_size 1 \
    --nodes 1 \
    --node_rank 0

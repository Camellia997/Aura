#!/bin/bash
# =====================================================================
# Single-node multi-GPU inference (sequence parallel over GPUS_PER_NODE)
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
SAVE_DIR="results/single_node"

# ---------------------------------------------------------------------
# Generation / parallel settings
# ---------------------------------------------------------------------
RESOLUTION="720*1280"   # 720P vertical: 720*1280, horizontal: 1280*720
GEN_MODE="ref"
BOUNDARY=0.875
GPUS_PER_NODE=8         # must match --ulysses_size below (cfg.num_heads must be divisible by it)

torchrun --nproc_per_node=$GPUS_PER_NODE inference.py \
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
    --sample_shift 12.0 \
    --offload_model True \
    --dit_fsdp \
    --t5_fsdp \
    --ulysses_size $GPUS_PER_NODE \
    --apply_rope_in_selfattn False \
    --nodes 1 \
    --node_rank 0

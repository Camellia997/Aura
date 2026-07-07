#!/bin/bash
# =====================================================================
# Multi-node multi-GPU inference.
# Each node runs sequence parallel over its GPUs (ulysses_size = GPUS_PER_NODE),
# and the validation set is sharded across nodes (--nodes / --node_rank).
#
# Launch this script on EVERY node, passing the same MASTER_ADDR/MASTER_PORT
# and NNODES, but a unique NODE_RANK per node (0..NNODES-1).
#
#   NNODES=4 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 bash jobs/infer_multi_node.sh
#   NNODES=4 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 bash jobs/infer_multi_node.sh
#   ...
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
SAVE_DIR="results/multi_node"

# ---------------------------------------------------------------------
# Distributed rendezvous (override via environment variables)
# ---------------------------------------------------------------------
NNODES=${NNODES:-2}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}   # must match --ulysses_size below

# ---------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------
RESOLUTION="720*1280"   # 720P vertical: 720*1280, horizontal: 1280*720
GEN_MODE="ref"
BOUNDARY=0.875

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv-conf timeout=1800 \
    inference.py \
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
    --nodes $NNODES \
    --node_rank $NODE_RANK

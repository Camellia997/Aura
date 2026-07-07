# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch

from src.distributed.fsdp import shard_model
from src import (
    HYVideoEditV1_high, 
    HYVideoEditV1_low, 
)

DIT_MODELS = {
    "v1_h": HYVideoEditV1_high,
    "v1_l": HYVideoEditV1_low,
}

def build_dit_models(args):
    vlm_dir = getattr(args, "vlm_dir", None)

    logging.info(f" >>> Building low noise model of version {args.dit_version_low}")
    low_noise_model = DIT_MODELS[args.dit_version_low](
        model_type="t2v", 
        patch_size=(1, 2, 2), 
        text_len=512, 
        in_dim=16, 
        dim=5120, 
        ffn_dim=13824, 
        freq_dim=256, 
        text_dim=4096, 
        out_dim=16, 
        num_heads=40, 
        num_layers=40, 
        window_size=(-1, -1), 
        qk_norm=True, 
        cross_attn_norm=True, 
        eps=1e-6, 
        apply_rope_in_selfattn=args.apply_rope_in_selfattn, 
        is_training=False, 
        vlm_dir=vlm_dir, 
    )

    logging.info(f" >>> Building high noise model of version {args.dit_version_high}")
    high_noise_model = DIT_MODELS[args.dit_version_high](
        model_type="t2v", 
        patch_size=(1, 2, 2), 
        text_len=512, 
        in_dim=16, 
        dim=5120, 
        ffn_dim=13824, 
        freq_dim=256, 
        text_dim=4096, 
        out_dim=16, 
        num_heads=40, 
        num_layers=40, 
        window_size=(-1, -1), 
        qk_norm=True, 
        cross_attn_norm=True, 
        eps=1e-6, 
        apply_rope_in_selfattn=args.apply_rope_in_selfattn, 
        is_training=False, 
        vlm_dir=vlm_dir, 
    )

    return high_noise_model, low_noise_model

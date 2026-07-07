# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import math
import functools
import random

import torch
import torch.cuda.amp as amp
from einops import rearrange
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from transformers import PretrainedConfig, Qwen2Config

from .attention import flash_attention
from src.modules.meta_query.qwen_vl_2_5 import QwenVL2_5_Encoder_v2_mtss
from src.modules.meta_query.transformer_encoder import Qwen2Encoder

from utils.communication import all_gather, all_to_all_4D
from utils.parallel_states import get_sequence_parallel_state, nccl_info

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2

from flash_attn_interface import flash_attn_3_cuda
def policy_fn(ctx, op, *args, **kwargs):
    if op is flash_attn_3_cuda.fwd.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE

checkpoint_context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)

T5_TEXT_LEN = 1024

def safe_tensor(x, clamp_value=1e4):
    """
    将 NaN/Inf 修复为安全值，只在检测到异常时才修复，不破坏正常梯度流
    """
    if not torch.is_floating_point(x):
        return x
    # 只在真正有 NaN/Inf 时才修复，避免无谓的计算和打印
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(f"⚠️⚠️⚠️ Warning: found NaN or Inf in input tensor (t5), replacing with safe values")
        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=clamp_value,
            neginf=-clamp_value,
        )
        x = torch.clamp(x, -clamp_value, clamp_value)
    return x


def build_mlp(hidden_size, projector_dim, align_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, align_dim),
            )


class MLLMInContextConfig(PretrainedConfig):
    model_type = "mllm-in-context"

    def __init__(
        self,
        mllm_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        diffusion_model_id: str = "Efficient-Large-Model/Sana_1600M_512px_diffusers",
        in_channels: int = 32,
        input_size: int = 32,
        num_metaqueries: int = 64,
        _gradient_checkpointing: bool = True,
        max_input_text_tokens: int = 256,
        connector_num_hidden_layers: int = 24,
        system_prompt: str = "You will be given an image or its caption. Please describe the content of the image in detail in your own words.",
        **kwargs,
    ):
        super().__init__()
        self.mllm_id = mllm_id
        self.diffusion_model_id = diffusion_model_id
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_metaqueries = num_metaqueries
        self._gradient_checkpointing = _gradient_checkpointing
        self.max_input_text_tokens = max_input_text_tokens
        self.connector_num_hidden_layers = connector_num_hidden_layers
        self.system_prompt = system_prompt


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        if get_sequence_parallel_state():
            x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
                s, n, -1, 2))
        else:
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
                seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        if get_sequence_parallel_state():
            sp_size = nccl_info.sp_size
            sp_rank = nccl_info.rank_within_group
            freqs_i = pad_freqs(freqs_i, s * sp_size)
            s_per_rank = s
            freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) * s_per_rank), :, :]
            x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
            x_i = torch.cat([x_i, x[i, s:]])
        else:
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype)

@torch.amp.autocast('cuda', enabled=False)
def shift_rope(x, grid_sizes, freqs, attn_dim, 
               shift_f: bool, 
               shift_h: bool, 
               shift_w: bool, 
               shift_f_size: int,
               shift_h_size: int, 
               shift_w_size: int):
    """Shift rotary embeddings. The shifting could be applyed at either F, H, or W dimension. 
    Args:
        x(Tensor): 
        grid_sizes(List[Tensor]): List of grid sizes, each item is in format [patch_f, patch_h, patch_w]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads 2]
    """
    # s, c = x.size(1), attn_dim // 2 # not used
    c = attn_dim // 2

    # split freqs
    ori_freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    f, h, w = grid_sizes.tolist()[0]
    seq_len = f * h * w
    
    if shift_f:
        freqs_f = ori_freqs[0][shift_f_size : shift_f_size + f].view(f, 1, 1, -1).expand(f, h, w, -1)
    else:
        freqs_f = ori_freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1)
    if shift_h:
        freqs_h = ori_freqs[1][shift_h_size : shift_h_size + h].view(1, h, 1, -1).expand(f, h, w, -1)
    else:
        freqs_h = ori_freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
    if shift_w:
        freqs_w = ori_freqs[2][shift_w_size : shift_w_size + w].view(1, 1, w, -1).expand(f, h, w, -1)
    else:
        freqs_w = ori_freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)

    shifted_freqs = torch.cat((freqs_f, freqs_h, freqs_w), dim=-1).reshape(seq_len, 1, -1)

    return shifted_freqs

@torch.amp.autocast('cuda', enabled=False)
def direct_rope_apply(x, freqs, unsqueeze=True):
    s, n, c = x.size(1), x.size(2), x.size(3) // 2

    x_i = torch.view_as_complex(x.to(torch.float64).reshape(s, n, -1, 2))
    x_i = torch.view_as_real(x_i * freqs).flatten(2)

    if unsqueeze:
        x_i = x_i.unsqueeze(0)

    return x_i.to(x)

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6, 
                 apply_rope=True):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.apply_rope = apply_rope

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if self.apply_rope:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
        else:
            q = direct_rope_apply(q, freqs, unsqueeze=True)
            k = direct_rope_apply(k, freqs, unsqueeze=True)

        if get_sequence_parallel_state():
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, scatter_dim=2, gather_dim=1)
        
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        if get_sequence_parallel_state():
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, vlm_context=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if vlm_context is not None:
            k_vlm = self.norm_k(self.k(vlm_context)).view(b, -1, n, d)
            v_vlm = self.v(vlm_context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=None)

        if vlm_context is not None:
            x_vlm = flash_attention(q, k_vlm, v_vlm, k_lens=None)
            x_vlm = x_vlm.flatten(2)

        # output
        x = x.flatten(2)

        if vlm_context is None:
            x = self.o(x)
        else:
            x = self.o(x / 2 + x_vlm / 2)
            # print(f" ------> x: {x.min().item()} {x.max().item()} | x_vlm: {x_vlm.min().item()} {x_vlm.max().item()}")
            # x = self.o(x_vlm)
        return x


class WanIPCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_vlm = nn.Linear(dim, dim)    # initialized from pretrained self.k
        self.v_vlm = nn.Linear(dim, dim)    # initialized from pretrained self.k
        self.o_vlm = nn.Linear(dim, dim)    # for weight, initialized as identity, for bias, initialized as zeros
        
        self.norm_k_vlm = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, vlm_context_emb):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            vlm_context_emb(Tensor): Shape [B, Lq*n, C]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention (text)
        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)

        # compute attention (vlm)
        if vlm_context_emb is not None:
            k_vlm = self.norm_k_vlm(self.k_vlm(vlm_context_emb)).view(b, -1, n, d)
            v_vlm = self.v_vlm(vlm_context_emb).view(b, -1, n, d)
            x_vlm = flash_attention(q, k_vlm, v_vlm, k_lens=None)
            x_vlm = x_vlm.flatten(2)
            x_vlm = self.o_vlm(x_vlm)

        # output
        if vlm_context_emb is not None:
            x = self.o(x + x_vlm)
        else:
            x = self.o(x)

        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim,
                                            num_heads,
                                            (-1, -1),
                                            qk_norm,
                                            eps)
        
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        **kwargs, 
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        if len(e.shape) == 3:
            with amp.autocast(dtype=torch.float32):
                e = (self.modulation + e).chunk(6, dim=1)
            assert e[0].dtype == torch.float32

            # self-attention
            e_2 = [e[i].to(dtype=x.dtype, device=x.device) for i in range(6)]
            y = self.self_attn(
                self.norm1(x) * (1 + e_2[1]) + e_2[0], seq_lens, grid_sizes,
                freqs)
            # with amp.autocast(dtype=torch.float32):
            x = x + y * e_2[2]

            # cross-attention & ffn function
            x_norm = self.norm3(x)
            x = x + self.cross_attn(x_norm, context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e_2[4]) + e_2[3])
            with amp.autocast(dtype=e_2[0].dtype):
                x = x + y * e_2[5]
        elif len(e.shape) == 4:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            assert e[0].dtype == torch.float32

            ori_dtpye = x.dtype
            # self-attention
            # y = self.self_attn(
            #     self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            #     seq_lens, grid_sizes, freqs)
            y = self.self_attn(
                (self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)).to(ori_dtpye),
                seq_lens, grid_sizes, freqs)

            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[2].squeeze(2)

            x = x.to(ori_dtpye)
            # cross-attention & ffn function
            def cross_attn_ffn(x, context, context_lens, e):
                x = x + self.cross_attn(self.norm3(x), context, context_lens)
                y = self.ffn(
                    (self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)).to(ori_dtpye))
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    x = x + y * e[5].squeeze(2)
                return x

            x = cross_attn_ffn(x, context, context_lens, e)
            x = x.to(ori_dtpye)
        else:
            raise ValueError
        return x


class WanIPAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6, 
                 apply_rope_in_selfattn=True):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.apply_rope_in_selfattn = apply_rope_in_selfattn

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, apply_rope=self.apply_rope_in_selfattn)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim,
                                            num_heads,
                                            (-1, -1),
                                            qk_norm,
                                            eps)
        
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        vlm_context=None,
        **kwargs, 
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            vlm_context_emb(Tensor): Shape [B, Lq*n, C]
            face_mask(Tensor): Shape [B, n, L1]
        """
        assert e.dtype == torch.float32
        if len(e.shape) == 3:
            with amp.autocast(dtype=torch.float32):
                e = (self.modulation + e).chunk(6, dim=1)
            assert e[0].dtype == torch.float32

            # self-attention
            e_2 = [e[i].to(dtype=x.dtype, device=x.device) for i in range(6)]
            y = self.self_attn(
                self.norm1(x) * (1 + e_2[1]) + e_2[0], seq_lens, grid_sizes,
                freqs, **kwargs)
            # with amp.autocast(dtype=torch.float32):
            x = x + y * e_2[2]

            # cross-attention & ffn function
            x_norm = self.norm3(x)
            x_cross = self.cross_attn(x_norm, context, context_lens)
            x = x + x_cross
            y = self.ffn(self.norm2(x) * (1 + e_2[4]) + e_2[3])
            with amp.autocast(dtype=e_2[0].dtype):
                x = x + y * e_2[5]
        elif len(e.shape) == 4:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            assert e[0].dtype == torch.float32

            ori_dtpye = x.dtype
            # self-attention
            y = self.self_attn(
                (self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)).to(ori_dtpye),
                seq_lens, grid_sizes, freqs, **kwargs)

            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[2].squeeze(2)

            x = x.to(ori_dtpye)
            # cross-attention & ffn function
            def cross_attn_ffn(x, context, context_lens, e, vlm_context=None):
                x_cross = self.cross_attn(self.norm3(x), context, context_lens, vlm_context)
                x = x + x_cross
                y = self.ffn(
                    (self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)).to(ori_dtpye))
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    x = x + y * e[5].squeeze(2)
                return x

            x = cross_attn_ffn(x, context, context_lens, e, vlm_context)
            x = x.to(ori_dtpye)
        else:
            raise ValueError
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        if len(e.shape) == 2:
            with amp.autocast(dtype=torch.float32):
                e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
                x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        elif len(e.shape) == 3:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
                x = (
                    self.head(
                        self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))

        return x


class SteepSigmoid(nn.Module):
    def __init__(self, k=5.0):
        super(SteepSigmoid, self).__init__()
        self.k = k # 你甚至可以将 k 设为可学习的参数: nn.Parameter(torch.tensor(k))

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.k * x))


class Head_mask(nn.Module):
    def __init__(self, dim, out_dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.eps = eps

        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)
        
        # act
        self.act = nn.Sigmoid()
        # self.act = SteepSigmoid(k=0.5)
    
    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
            # e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)    # v6
            # x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))         # v6
            x = self.act(x)
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), 
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), 
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class AlignmentQFormer(nn.Module):
    def __init__(self, 
                 t5_dim, 
                 vlm_dim, 
                 in_dim, 
                 out_dim, 
                 ffn_dim,
                 num_heads, 
                 num_layers, 
                 num_queries, 
                 qk_norm=True,
                 eps=1e-6,):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        self.proj = nn.Linear(t5_dim, in_dim)
        # self.t5_linear = nn.Linear(t5_dim, in_dim)
        self.t5_norm = WanLayerNorm(in_dim, eps)
        # self.vlm_linear = nn.Linear(vlm_dim, in_dim)
        self.vlm_norm = WanLayerNorm(in_dim, eps)

        self.query = nn.Parameter(torch.randn(num_queries, in_dim), requires_grad=True)
        self.cross_blocks = nn.ModuleList([
            WanCrossAttention(in_dim, num_heads, (-1, -1), qk_norm, eps) for _ in range(num_layers)
        ])

        self.ffn = nn.Sequential(
                nn.Linear(in_dim, ffn_dim), 
                nn.GELU(approximate="tanh"), 
                nn.Linear(ffn_dim, out_dim)
            )

        # self._initialize_query()

    def _initialize_query(self):
        """
        正交初始化 query
        """
        # torch.nn.init.orthogonal_(self.query, gain=0.5)
        """手动正交初始化（不强制范数为 1）"""
        # 先随机正态初始化
        torch.nn.init.normal_(self.query, mean=0.0, std=1.0)
        N, D = self.query.shape
        # SVD 分解获取正交矩阵
        U, S, V = torch.svd(self.query)
        # 选择正交矩阵（保证行正交）
        orthogonal_mat = V.t()
        # 计算随机初始化的queyr的模
        init_norm = torch.linalg.norm(self.query, dim=1)
        # 恢复到原始的模
        orthogonal_mat = orthogonal_mat * init_norm.unsqueeze(dim=1)
        # 赋值（此时 orthogonal_mat 行向量正交，范数任意）
        self.query.data.copy_(orthogonal_mat)

    def forward(
        self, 
        t5_embed, 
        t5_attention_mask, 
        vlm_embed, 
    ):
        r"""
        Args:
            t5_embed(Tensor): Shape [B, L1, C]
            t5_attention_mask(Tensor): Shape [B, L1, C]
            vlm_embed(Tensor): Shape [B, L2, C]
        """

        b, l, c = t5_embed.shape
        b_vlm, l, c = vlm_embed.shape

        # t5_embed = self.t5_norm(self.t5_linear(t5_embed))
        # vlm_embed = self.vlm_norm(self.vlm_linear(vlm_embed))
        t5_embed = self.t5_norm(self.proj(t5_embed))
        vlm_embed = self.vlm_norm(self.proj(vlm_embed))

        t5_query = self.query.unsqueeze(dim=0).repeat(b, 1, 1)
        vlm_query = self.query.unsqueeze(dim=0).repeat(b_vlm, 1, 1)
        for block in self.cross_blocks:
            t5_query = block(t5_query, t5_embed, None)
            vlm_query = block(vlm_query, vlm_embed, None)

        t5_out = self.ffn(t5_query)
        vlm_out = self.ffn(vlm_query)

        return t5_out, vlm_out

    def get_normalized_orthogonal_query(self):
        """
        获取归一化后的正交 query 向量（核心接口：返回满足需求的向量组）
        :return: normalized_query - 形状 (num_queries, in_dim)，每个向量范数为 1，两两正交
        """
        # 对每个 query 向量单独做 L2 归一化（dim=1：对每行（单个 query）做归一化）
        normalized_query = torch.nn.functional.normalize(self.query.float(), p=2, dim=1)
        return normalized_query

    def calc_orthogonal_regularizatioin(self):
        """
        计算 query 的正交性惩罚损失
        :param weight: 正交性损失的权重，可根据任务调整（如 0.1, 1.0）
        :return: 正交性损失值（标量）
        """
        # 1. 计算 query 矩阵的 Gram 矩阵 (num_queries, num_queries)
        num_queries = self.query.size(0)
        normalized_query = self.get_normalized_orthogonal_query()
        query_gram = torch.matmul(normalized_query, normalized_query.t())
        # 2. 生成单位矩阵（与 Gram 矩阵同形状、同设备）
        identity = torch.eye(num_queries, device=query_gram.device).to(dtype=torch.float32)
        # 3. 计算 Gram 矩阵与单位矩阵的 L2 误差（作为正交性惩罚）
        ortho_loss = torch.nn.functional.mse_loss(query_gram, identity)
        # 4. 乘以权重后返回
        return ortho_loss


class REPAProj(nn.Module):
    def __init__(self, dim, out_dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.eps = eps

        self.norm = WanLayerNorm(dim, eps)
        self.head = build_mlp(dim, 2048, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6, 
                 apply_rope_in_selfattn=True, 
                 **kwargs, ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.apply_rope = True
        self.global_step = 0
        self.apply_rope_in_selfattn = apply_rope_in_selfattn
        self.is_training = kwargs.get("is_training", True)

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # first/last frame embedding, used for encoding first/last frame
        self.flf_embedding = nn.Conv3d(
            16, dim, kernel_size=patch_size, stride=patch_size)

        # ref embeddings, used for encoding reference images
        self.ref_embedding = nn.Conv3d(
            16, dim, kernel_size=patch_size, stride=patch_size)

        # video embeddings, used for encoding source video
        self.video_embedding = nn.Conv3d(
            16, dim, kernel_size=patch_size, stride=patch_size)
        
        # meta-query
        vlm_dir = kwargs.get("vlm_dir", None) or "Qwen/Qwen2.5-VL-3B-Instruct"
        self.vlm_metaquery = QwenVL2_5_Encoder_v2_mtss(
            vlm_dir, 
            max_edge=384, 
            max_aspect_ratio=1.75, 
            device=kwargs.get("device", torch.device("cuda")), 
            dtype=kwargs.get("dtype", torch.float32)
        )

        # query transformer encoder
        meta_query_config = MLLMInContextConfig(connector_num_hidden_layers=8)
        self.vlm_encoder = Qwen2Encoder(
            Qwen2Config(
                hidden_size=self.vlm_metaquery.model.config.hidden_size,
                intermediate_size=self.vlm_metaquery.model.config.hidden_size * 4,
                num_hidden_layers=meta_query_config.connector_num_hidden_layers,
                num_attention_heads=self.vlm_metaquery.model.config.hidden_size // 64,
                num_key_value_heads=self.vlm_metaquery.model.config.hidden_size // 64,
                initializer_range=0.014,
                use_cache=False,
                rope=True,
                qk_norm=True,
            ),
        )
        self.vlm_connector = nn.Sequential(
            nn.Linear(self.vlm_metaquery.model.config.hidden_size, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.vlm_connector_proj = nn.Linear(dim, dim, bias=False)

        self.align_qformer = AlignmentQFormer(
            # t5_dim=4096, 
            t5_dim=dim, 
            vlm_dim=self.vlm_metaquery.model.config.hidden_size, 
            in_dim=dim, 
            out_dim=768,        # default=768
            ffn_dim=ffn_dim,
            num_heads=40, 
            num_layers=2,       # default=2 
            num_queries=32,     # default=32
            qk_norm=True,
            eps=1e-6,
        )

        # learnable tokens
        self.learnable_tokens = nn.ParameterDict(
            {
                "ff": nn.Parameter(torch.zeros(1, dim), requires_grad=True),
                "lf": nn.Parameter(torch.zeros(1, dim), requires_grad=True),
                "ref": nn.Parameter(torch.zeros(1, dim), requires_grad=True),
                "vid": nn.Parameter(torch.zeros(1, dim), requires_grad=True),
                "human": nn.Parameter(torch.zeros(1, dim), requires_grad=True),
                "object": nn.Parameter(torch.zeros(1, dim), requires_grad=True),
                "scene": nn.Parameter(torch.zeros(1, dim), requires_grad=True),
                "memory": nn.Parameter(torch.zeros(1, dim), requires_grad=True),
            }
        )

        self.memory_tokens = nn.Parameter(torch.zeros(1, 16), requires_grad=True)

        # text embeddings
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        # Learnable logit scale for CLIP loss (will be loaded from pretrained and frozen)
        if kwargs.get("is_training", True):
            # self.logit_scale = nn.Parameter(
            #     torch.ones([]) * math.log(1 / 0.07)
            # )
            pass    # TODO: to modify it as 1d conv
        
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanIPAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                window_size, qk_norm, cross_attn_norm, eps, 
                                apply_rope_in_selfattn=apply_rope_in_selfattn)
            # WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
            #                     window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        self.align_blocks = []     # 特征对齐层
        # self.align_block_projs = nn.ModuleDict(
        #     {
        #         str(idx): build_mlp(dim, 2048, 768) for idx in range(len(self.blocks))
        #     }
        # )

        align_dim = 1408                    # 1024 for vitl, 1408 for vitg
        projector_dim = 2048
        self.align_layers = [16]            # 默认加在16层
        self.repa_blocks = nn.ModuleDict(
            {
                str(idx): REPAProj(dim, align_dim) for idx in self.align_layers
            }
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.enable_teacache = False
        self.init_weights()

        self.trd_loss_fn = TRDLoss(margin=0.1, lambda_temporal=1.0)

    def update_align_blocks(self):
        """每一步训练前都更新align_block的层数，这样每次只用对齐一层，循环对齐
        """
        num_blocks = len(self.blocks)
        align_block = self.global_step % num_blocks
        self.align_blocks = [align_block]

    def apply_learnable_tokens(self, context_embs, context_type="ff", ref_id=None):
        """
        Args:
            context_embs: Shape (b c t h w)
        """
        b, c, t, h, w = context_embs.shape
        if context_type in ["ff", "lr"]:
            token = rearrange(self.learnable_tokens[context_type], 
                              "b (t h w c) -> b c t h w", b=1, t=1, h=1, w=1).repeat(b, 1, t, h, w)
        elif context_type == "ref":
            # 🔥 FIX: 始终让所有 ref learnable tokens 参与计算，避免 DDP 梯度不一致
            # 根据 ref_id 选择主要的 token，但其他 token 也会参与（权重为 0）
            if ref_id is not None:
                if ref_id < 100:
                    c_type = "human"
                elif ref_id < 200:
                    c_type = "object"
                elif ref_id < 300:
                    c_type = "scene"
                else:
                    c_type = "memory"
                # 主要 token
                token = rearrange(self.learnable_tokens[context_type] + self.learnable_tokens[c_type], 
                                "b (t h w c) -> b c t h w", b=1, t=1, h=1, w=1).repeat(b, 1, t, h, w)
                # 让其他 token 也参与计算（但权重为 0，不影响结果）
                for other_type in ["human", "object", "scene", "memory"]:
                    if other_type != c_type:
                        other_token = rearrange(self.learnable_tokens[other_type], 
                                              "b (t h w c) -> b c t h w", b=1, t=1, h=1, w=1).repeat(b, 1, t, h, w)
                        token = token + other_token * 0.0  # 权重为 0，但参与计算图
            else:
                token = rearrange(self.learnable_tokens[context_type] + self.learnable_tokens["human"], 
                                "b (t h w c) -> b c t h w", b=1, t=1, h=1, w=1).repeat(b, 1, t, h, w)
                # 让其他 token 也参与计算
                for other_type in ["object", "scene", "memory"]:
                    other_token = rearrange(self.learnable_tokens[other_type], 
                                          "b (t h w c) -> b c t h w", b=1, t=1, h=1, w=1).repeat(b, 1, t, h, w)
                    token = token + other_token * 0.0
                    
        elif context_type == "vid":
            token = rearrange(self.learnable_tokens[context_type], 
                              "b (t h w c) -> b c t h w", b=1, t=1, h=1, w=1).repeat(b, 1, t, h, w)
        return context_embs + token

    def reshape_hidden_states_to_grid(
        self, 
        hidden_states, 
        grid_size, 
        pre_context=None, 
        pre_seq_lens=0, 
        post_context=None, 
        post_seq_lens=0
    ):  
        x = hidden_states.clone()
        if pre_context is not None:
            x = x[:, pre_seq_lens:]
        if post_context is not None:
            x = x[:, :-post_seq_lens]

        _, f, h, w = grid_size[0]
        x = rearrange(x, "b (f h w) c -> b c f h w", f=f, h=h//2, w=w//2)
        return x

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        cond_flag=False, 
        **input_kwargs, 
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # Get Global Rank (unique across all nodes)
        global_rank = int(os.environ.get("RANK", 0))
        # Get Local Rank (unique within the current node)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # ----- 开启以更新对齐层 -----
        # self.update_align_blocks()
        
        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # --- encode context ---
        # t5 text context
        if self.is_training:
            text_context = context["comp_text_context"]
            text_attention_mask = context["comp_attention_mask"]
        else:
            text_context = context["text_context"]
            text_attention_mask = context["text_attention_mask"]
        text_context_lens = None
        text_context_emb = self.text_embedding(
            torch.stack([
                torch.cat(
                    # [u, u.new_zeros(max(0, self.text_len - u.size(0)), u.size(1))])
                    [u, u.new_zeros(max(0, T5_TEXT_LEN - u.size(0)), u.size(1))])
                for u in text_context
            ]))

        # meta-query
        ref_id = context["ref_id"]

        if self.is_training:
            # print(f" -----> Start META-QUERY")
            vlm_context_emb, vlm_attention_mask = self.vlm_metaquery(
                video_prompts=context["video_prompt"], 
                context=context, 
                padding_info=None, 
            )
        
        # query-encoder & connector
        if self.is_training:
            # print(f" -----> Start VLM-ENCODER")
            vlm_context_emb = self.vlm_encoder(vlm_context_emb)
            vlm_context_emb = self.vlm_connector(vlm_context_emb)
            vlm_context_emb = safe_tensor(vlm_context_emb)
        else:
            vlm_context_emb = context["vlm_context_emb"]
        
        # calculate alignment losses
        loss_dict = None
        if input_kwargs.get("return_loss", False) and input_kwargs.get("alignment_enabled", False):
            loss_dict = self.calc_alignment_losses(
                vlm_context_emb=vlm_context_emb, 
                vlm_attention_mask=vlm_attention_mask, 
                text_context_emb=text_context_emb, 
                text_attention_mask=text_attention_mask, 
                neg_text_context=context["neg_comp_text_context"], 
                neg_text_attention_mask=context["neg_comp_attention_mask"], 
                t5_text_len=T5_TEXT_LEN,
            )

        vlm_context_emb_proj = self.vlm_connector_proj(vlm_context_emb)

        # get input noise shape
        x_shape = [u.shape for u in x]
        num_valid_context = 0

        # first frame context
        if isinstance(context, dict) and "ff_context" in context:
            ff_context = context["ff_context"]      # (b c 1 h w)
            ff_seq_lens = ff_context.size(2) * ff_context.size(3) * ff_context.size(4) // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        else:
            ff_context = None
            ff_seq_lens = 0

        # last frame context
        if isinstance(context, dict) and "lf_context" in context:
            lf_context = context["lf_context"]      # (b c 1 h w)
            lf_seq_lens = lf_context.size(2) * lf_context.size(3) * lf_context.size(4) // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        else:
            lf_context = None
            lf_seq_lens = 0

        # reference context
        if isinstance(context, dict) and "ref_context" in context:
            ref_context = context["ref_context"]    # (b*n c 1 h w)
            ref_seq_lens = ref_context.size(0) * ref_context.size(3) * ref_context.size(4) // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
            num_valid_context += ref_context.size(0)
        else:
            ref_context = None
            ref_seq_lens = 0

        # video context
        if isinstance(context, dict) and "video_context" in context:
            video_context = context["video_context"]    # (b c n h w)
            video_seq_lens = video_context.size(2) * video_context.size(3) * video_context.size(4) // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        else:
            video_context = None
            video_seq_lens = 0

        # (Optional) memory token
        if ref_context is not None and input_kwargs.get("use_memory_token", True):
            h_, w_ = x.shape[-2:]
            num_memory_context = 6 - num_valid_context
            memory_context = self.memory_tokens.repeat(h_*w_*num_memory_context, 1).unsqueeze(0) # (b f c)
            memory_context = rearrange(memory_context, "b (f h w) c -> (b f) c 1 h w", f=num_memory_context, h=h_, w=w_)
            ref_context = torch.concat([ref_context, memory_context], dim=0)
            ref_seq_lens += num_memory_context * h_ * w_ // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
            ref_id = torch.cat([ref_id, torch.tensor([300] * num_memory_context).to(ref_id).view(1, -1)], dim=1)

        # embeddings
        cat_x = []
        if ff_context is not None:
            ff_context_emb = [self.apply_learnable_tokens(
                self.flf_embedding(u.unsqueeze(0)), context_type="ff"
            ) for u in ff_context]
            ff_grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in ff_context_emb])
            cat_x += ff_context_emb
        else:
            ff_context_emb = None
            ff_grid_sizes = None

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        cat_x += x
        
        if lf_context is not None:
            lf_context_emb = [self.apply_learnable_tokens(
                self.flf_embedding(u.unsqueeze(0)), context_type="lf"
            ) for u in lf_context]
            lf_grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in lf_context_emb])
            cat_x += lf_context_emb
        else:
            lf_context_emb = None
            lf_grid_sizes = None
        
        if ref_context is not None:
            if ref_id is None:
                ref_context_emb = [self.apply_learnable_tokens(
                    self.ref_embedding(u.unsqueeze(0)), context_type="ref"
                ) for u in ref_context]
                ref_grid_sizes = torch.stack(
                    [torch.tensor(u.shape[2:], dtype=torch.long) for u in ref_context_emb])
            else:
                ref_context_emb = [self.apply_learnable_tokens(
                    self.ref_embedding(u.unsqueeze(0)), context_type="ref", ref_id=r.item()
                ) for u, r in zip(ref_context, ref_id.view(-1))]
                ref_grid_sizes = torch.stack(
                    [torch.tensor(u.shape[2:], dtype=torch.long) for u in ref_context_emb])
            cat_x += ref_context_emb
        else:
            ref_id = None
            ref_context_emb = None
            ref_grid_sizes = None
        
        if video_context is not None:
            video_context_emb = [self.apply_learnable_tokens(
                self.video_embedding(u.unsqueeze(0)), context_type="vid"
            ) for u in video_context]
            video_grid_sizes = torch.stack(
                [torch.tensor(u.shape[2:], dtype=torch.long) for u in video_context_emb])
            cat_x += video_context_emb
        else:
            video_context_emb = None
            video_grid_sizes = None

        # --- compute the shifted rotary embedding for noise latents and context latents ---
        freqs = self.gen_rope_embeddings(ref_id, ff_context_emb, ff_grid_sizes, 
                                         x, grid_sizes, 
                                         lf_context_emb, lf_grid_sizes, 
                                         ref_context_emb, ref_grid_sizes, 
                                         video_context_emb, video_grid_sizes, 
                                         f_shift=5)

        # -----------------------------------------------
        x = torch.cat(cat_x, dim=2)

        grid_sizes = torch.stack([torch.tensor(x.shape[2:])])
        x = [x.flatten(2).transpose(1, 2)]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if seq_len != 0:
            seq_len = seq_lens.max().item()
            assert seq_lens.max() <= seq_len
            x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                          dim=1) for u in x
            ])
        
        # time embeddings
        if t.dim() == 1:
            t = t.view(-1, 1).repeat(t.size(0), seq_len)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            # set the timestep according to ref and video context as zero
            t[:ff_seq_lens] = 0.0
            if lf_seq_lens + ref_seq_lens + video_seq_lens != 0:
                t[-(lf_seq_lens + ref_seq_lens + video_seq_lens):] = 0.0
            # get timestep embedding
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))

            assert e.dtype == torch.float32

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        if get_sequence_parallel_state():
            x = torch.chunk(x, nccl_info.sp_size, dim=1)[nccl_info.rank_within_group]
            freqs = torch.chunk(freqs, nccl_info.sp_size, dim=0)[nccl_info.rank_within_group]
            e = torch.chunk(e, nccl_info.sp_size, dim=1)[nccl_info.rank_within_group]
            e0 = torch.chunk(e0, nccl_info.sp_size, dim=1)[nccl_info.rank_within_group]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs if self.apply_rope_in_selfattn else freqs,
            context=text_context_emb,
            context_lens=256, 
            vlm_context=vlm_context_emb_proj,   # [DEBUG]
            )

        if self.enable_teacache:
            if cond_flag:
                modulated_inp = e
                if self.cnt == 0 or self.cnt == self.num_steps-1:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
                else:
                    rescale_func = np.poly1d(self.coefficients)
                    if cond_flag:
                        self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                    if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.accumulated_rel_l1_distance = 0
                self.previous_modulated_input = modulated_inp
                self.cnt = 0 if self.cnt == self.num_steps-1 else self.cnt + 1
                self.should_calc = should_calc
            else:
                should_calc = self.should_calc
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        align_hidden_states = list()
        align_hidden_states_tuple = list()
        self.grid_sizes = x_shape
        if self.enable_teacache:
            if not should_calc:
                x = x + self.previous_residual_cond if cond_flag else x + self.previous_residual_uncond
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                if cond_flag:
                    self.previous_residual_cond = x - ori_x
                else:
                    self.previous_residual_uncond = x - ori_x
        else:
            for block_id, block in enumerate(self.blocks):
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, e0, seq_lens, grid_sizes, 
                            self.freqs if self.apply_rope_in_selfattn else freqs,
                            text_context_emb, 256,
                            vlm_context_emb_proj,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, e0, seq_lens, grid_sizes, 
                        self.freqs if self.apply_rope_in_selfattn else freqs,
                        text_context_emb, 256,
                        vlm_context_emb_proj,
                        use_reentrant=False,
                        context_fn=checkpoint_context_fn,
                    )
                else:
                    x = block(x, **kwargs)

                # ----- 计算REPA特征（TRD_loss）-----
                if input_kwargs.get("return_loss", False) and input_kwargs.get("repa_enabled", False) and block_id in self.align_layers:
                    align_hidden_states.append(self.repa_blocks[str(block_id)](x, e))

        # head
        x = self.head(x, e)

        if get_sequence_parallel_state():
            x = all_gather(x, dim=1).contiguous()
            for block_id in range(len(align_hidden_states)):
                align_hidden_states[block_id] = all_gather(align_hidden_states[block_id], dim=1).contiguous()
                if ff_context is not None:
                    align_hidden_states[block_id] = align_hidden_states[block_id][:, ff_seq_lens:]
                if ref_context is not None or lf_context is not None or video_context is not None:
                    align_hidden_states[block_id] = align_hidden_states[block_id][:, :-(ref_seq_lens + lf_seq_lens + video_seq_lens)]
        else:
            for block_id in range(len(align_hidden_states)):
                # self.align_hidden_states[str(block_id)] = all_gather(self.align_hidden_states[str(block_id)], dim=1).contiguous()
                if ff_context is not None:
                    align_hidden_states[block_id] = align_hidden_states[block_id][:, ff_sel_lens:]
                if ref_context is not None or lf_context is not None or video_context is not None:
                    align_hidden_states[block_id] = align_hidden_states[block_id][:, :-(ref_seq_lens + lf_seq_lens + video_seq_lens)]

        if input_kwargs.get("return_loss", False) and context.get("repa_align_target", None) is not None:
            repa_loss = self.calc_repa_losses(
                align_source=align_hidden_states, 
                align_target=context["repa_align_target"], 
                grid_sizes=self.grid_sizes[0])
            if loss_dict is None:
                loss_dict = {"repa_loss": repa_loss}
            else:
                loss_dict["repa_loss"] = repa_loss

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        
        if ff_context is not None:
            x = [u[:, v.shape[1]:] for u, v in zip(x, ff_context)]

        x = [u[:, :v[1]] for u, v in zip(x, x_shape)]

        # ----- 更新global_step -----
        self.global_step += 1

        if input_kwargs.get("return_loss", False):
            return [u.float() for u in x], loss_dict
        else:
            return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # # basic init
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init vlm-connector
        nn.init.zeros_(self.vlm_connector[2].weight)
        nn.init.zeros_(self.vlm_connector[2].bias)

        # init vlm-connector-proj
        init_weight = torch.eye(self.vlm_connector_proj.weight.size(0)) + 1e-5 * torch.randn_like(self.vlm_connector_proj.weight)
        self.vlm_connector_proj.weight = torch.nn.Parameter(init_weight, requires_grad=True)

        # for block_id in range(len(self.blocks)):
        #     # nn.init.zeros_(self.blocks[block_id].cross_attn.mask_head.head.weight)
        #     # nn.init.normal_(self.blocks[block_id].cross_attn.mask_head.head.weight, std=0.02)
        #     # nn.init.zeros_(self.blocks[block_id].cross_attn.mask_head.head.bias)
        #     # nn.init.constant_(self.blocks[block_id].cross_attn.mask_head.head.bias, -2.19)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def reorg_state_dict(self, input_state_dict):
        state_dict_mapping = {
            "patch_embedding.weight": [
                ("ref_embedding.weight", 16), 
                ("video_embedding.weight", 16), 
                ("flf_embedding.weight", 16), 
            ], 
            "patch_embedding.bias": [
                ("ref_embedding.bias", None), 
                ("video_embedding.bias", None), 
                ("flf_embedding.bias", None), 
            ]
        }

        cross_state_dict_mapping = {}
        for block_id in range(len(self.blocks)):
            cross_state_dict_mapping[f"blocks.{block_id}.cross_attn.k.weight"] = f"blocks.{block_id}.cross_attn.k_vlm.weight"
            cross_state_dict_mapping[f"blocks.{block_id}.cross_attn.k.bias"] = f"blocks.{block_id}.cross_attn.k_vlm.bias"
            cross_state_dict_mapping[f"blocks.{block_id}.cross_attn.v.weight"] = f"blocks.{block_id}.cross_attn.v_vlm.weight"
            cross_state_dict_mapping[f"blocks.{block_id}.cross_attn.v.bias"] = f"blocks.{block_id}.cross_attn.v_vlm.bias"
            cross_state_dict_mapping[f"blocks.{block_id}.cross_attn.o.weight"] = f"blocks.{block_id}.cross_attn.o_vlm.weight"
            cross_state_dict_mapping[f"blocks.{block_id}.cross_attn.o.bias"] = f"blocks.{block_id}.cross_attn.o_vlm.bias"
            cross_state_dict_mapping[f"blocks.{block_id}.cross_attn.norm_q.weight"] = f"blocks.{block_id}.cross_attn.norm_q_vlm.weight"
            cross_state_dict_mapping[f"blocks.{block_id}.cross_attn.norm_k.weight"] = f"blocks.{block_id}.cross_attn.norm_k_vlm.weight"

        output_state_dict = {}
        for key, val in input_state_dict.items():
            if key not in state_dict_mapping.keys():
                output_state_dict[key] = val
            else:
                output_state_dict[key] = val
                new_keys = state_dict_mapping[key]
                for (nkey, ndim) in new_keys:
                    if ndim is None:
                        output_state_dict[nkey] = val
                    else:
                        output_state_dict[nkey] = val[:, :ndim]

        for key, val in input_state_dict.items():
            if key in cross_state_dict_mapping.keys():
                new_key = cross_state_dict_mapping[key]
                output_state_dict[new_key] = val

        return output_state_dict

    def merge_vlm_embedding(self, text_context_emb, vlm_context_emb, text_attention_mask):
        t5_len = text_context_emb.size(1)
        vlm_len = vlm_context_emb.size(1)
        merged_context_emb = []

        for (temb, vemb, tmask) in zip(text_context_emb, vlm_context_emb, text_attention_mask):
            valid_text_len = int(tmask.sum().item())
            # merged_emb = torch.cat([temb[:valid_text_len], vemb], dim=0).unsqueeze(0)    # right padding without original t5 padding (1, l, c)
            merged_emb = torch.cat([vemb, temb], dim=0).unsqueeze(0) # left padding with original t5 padding kept
            merged_context_emb.append(merged_emb)
        merged_context_emb = torch.cat(merged_context_emb, dim=0)

        return merged_context_emb

    def setup_trainable_params(self, mode: str = "full"):
        if mode == "full":
            for name, param in self.named_parameters():
                if "vlm_metaquery" in name and "query_tokens" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif mode == "partial-full":
            for name, param in self.named_parameters():
                if "vlm_metaquery" in name:
                    param.requires_grad = False
                elif "align_qformer" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif mode == "dit-only":
            for name, param in self.named_parameters():
                if "vlm_metaquery" in name:
                    param.requires_grad = False
                elif "vlm_encoder" in name:
                    param.requires_grad = False
                elif "vlm_connector" in name and "vlm_connector_proj" not in name:
                    param.requires_grad = False
                # elif "vlm_connector_proj" in name:
                #     param.requires_grad = False
                elif "align_qformer" in name:
                    param.requires_grad = True  # This is wierd, but it works, LOL
                elif "logit_scale" in name:
                    param.requires_grad = False
                # elif "align_qformer" in name:       # DBEUG
                #     param.requires_grad = False
                elif "repa" in name:                # DEBUG
                    param.requires_grad = True
                elif "o_vlm" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = True
        elif mode == "fix-cross-attn":
            for name, param in self.named_parameters():
                if "vlm_metaquery" in name:
                    if "query_tokens" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                elif "vlm_encoder" in name:
                    param.requires_grad = True
                elif "vlm_connector" in name:
                    param.requires_grad = True
                elif "align_qformer" in name:
                    param.requires_grad = False  # This is wierd, but it works, LOL
                elif "logit_scale" in name:
                    param.requires_grad = False
                elif "repa" in name:                # DEBUG
                    param.requires_grad = False
                elif "cross_attn" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif mode == "meta-query":
            for name, param in self.named_parameters():
                if "vlm_metaquery" in name and "query_tokens" in name:
                    param.requires_grad = True
                elif "vlm_encoder" in name:
                    param.requires_grad = True
                elif "vlm_connector" in name:
                    param.requires_grad = True
                elif "vlm_connector_proj" in name:
                    param.requires_grad = True
                # elif "align_qformer" in name:
                #     param.requires_grad = True  
                elif "logit_scale" in name:
                    param.requires_grad = True
                elif name.startswith("blocks.") and "vlm" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def freeze_dit_weights(self):
        for name, param in self.named_parameters():
            if name.startswith("blocks.") and "img" not in name and "mask_head" not in name:
                param.requires_grad = False
            if name.startswith("text_embedding."):
                param.requires_grad = False
            if name.startswith("time_embedding."):
                param.requires_grad = False
            if name.startswith("time_projection."):
                param.requires_grad = False
            if name.startswith("head."):
                param.requires_grad = False
            if name.startswith("patch_embedding."):
                param.requires_grad = False
            if name.startswith("flf_embedding."):
                param.requires_grad = False
            if name.startswith("ref_embedding."):
                param.requires_grad = False

    def get_regularization_output(self):
        if self.align_hidden_states_tuple == {}:
            # return (self.align_hidden_states, self.align_targ, self.align_feat, self.grid_sizes)
            return (self.align_hidden_states, self.grid_sizes)
        else:
            # return (self.align_hidden_states, self.align_hidden_states_tuple, self.align_targ, self.align_feat, self.grid_sizes)
            return (self.align_hidden_states, self.align_hidden_states_tuple, self.grid_sizes)

    def calc_representation_regularization(self, inputs, target):
        # resize the target
        f, h, w = target.shape[2:]    # (b c f h w)

        if len(inputs) == 2:
            # hidden_states_dict, t5_emb, vlm_emb, grid_sizes = inputs
            hidden_states_dict, grid_sizes = inputs
            hidden_states_tuple_dict = None
        elif len(inputs) == 3:
            # hidden_states_dict, hidden_states_tuple_dict, t5_emb, vlm_emb, grid_sizes = inputs
            hidden_states_dict, hidden_states_tuple_dict, grid_sizes = inputs
        grid_sizes = list(grid_sizes[0])
        
        # # 1. calc the multi-modal semantic alignment loss between T5 and VLM
        # # 1.1 l1_loss between vlm_emb and t5_emb
        # mmsa_loss = torch.nn.functional.l1_loss(vlm_emb.float(), t5_emb.float()) * 10.0

        # # # 1.2 similarity between vlm_emb and t5_emb
        # # t5_gram = batch_cosine_similarity(t5_emb)
        # # vlm_gram = batch_cosine_similarity(vlm_emb)
        # # mmsa2_loss = torch.nn.functional.l1_loss(vlm_gram, t5_gram) * 10.0

        # 2. calc the REPA loss (TRD loss)
        target = rearrange(target.float(), "b c f h w -> b f (h w) c")

        trd_losses = 0
        for block_id, hidden_states in hidden_states_dict.items():
            hidden_states = rearrange(hidden_states, "b (f h w) c -> b c f h w", f=grid_sizes[1], h=grid_sizes[2]//2, w=grid_sizes[3]//2)
            hidden_states = torch.nn.functional.interpolate(
                hidden_states[:,:,1:].float(), 
                (f, h, w), mode="trilinear"
            )
            hidden_states = rearrange(hidden_states, "b c f h w -> b f (h w) c")
            trd_loss, _, _ = self.trd_loss_fn(hidden_states, target)
            trd_losses = trd_losses + trd_loss
        
        trd_losses = trd_losses / len(hidden_states_dict)

        # 3. (Optional) calc the cross-attn align loss (CAA loss)
        if hidden_states_tuple_dict is not None:
            caa_losses = 0
            for block_id, hidden_states_tuple in hidden_states_tuple_dict.items():
                h1 = hidden_states_tuple[0][:,:,1:]
                h2 = hidden_states_tuple[1][:,:,1:]
                _, _, f, h, w = h1.shape
                h1 = torch.nn.functional.interpolate(h1.float(), (f, h//2, w//2), mode="trilinear")
                h2 = torch.nn.functional.interpolate(h2.float(), (f, h//2, w//2), mode="trilinear")
                h1 = rearrange(h1, "b c f h w -> b f (h w) c")
                h2 = rearrange(h2, "b c f h w -> b f (h w) c")
                caa_loss, _, _ = self.trd_loss_fn(h1, h2)
                # caa_loss = torch.nn.functional.l1_loss(hidden_states_tuple[0], hidden_states_tuple[1])
                caa_losses = caa_losses + caa_loss * 10000
            caa_losses = caa_losses / len(hidden_states_tuple_dict)

        # losses = {"trd_loss": trd_losses, "mmsa_loss": mmsa_loss}
        losses = {"trd_loss": trd_losses}
        if hidden_states_tuple_dict is not None:
            losses["caa_loss"] = caa_losses
        return losses

    def contrastive_loss(self, anchor, positive, negative, margin=0.2):
        """
        Triplet Margin Loss with cosine distance.
        Args:
            anchor: VLM features (B, L, D) from align_qformer
            positive: T5 features (B, L, D) from same data
            negative: T5 features (B, L, D) from different data
            margin: margin value (default 0.2)
        Returns:
            loss: scalar tensor
        """
        # Mean pooling: (B, L, D) -> (B, D)
        anchor_pooled = anchor.mean(dim=1)
        positive_pooled = positive.mean(dim=1)
        negative_pooled = negative.mean(dim=1)

        # L2 normalize
        anchor_norm = torch.nn.functional.normalize(anchor_pooled.float(), p=2, dim=1)
        positive_norm = torch.nn.functional.normalize(positive_pooled.float(), p=2, dim=1)
        negative_norm = torch.nn.functional.normalize(negative_pooled.float(), p=2, dim=1)

        # Cosine similarity
        pos_sim = (anchor_norm * positive_norm).sum(dim=1)  # (B,)
        neg_sim = (anchor_norm * negative_norm).sum(dim=1)  # (B,)

        # Triplet margin loss: max(0, neg_sim - pos_sim + margin)
        # We want pos_sim > neg_sim + margin
        loss = torch.clamp(neg_sim - pos_sim + margin, min=0.0).mean()

        return loss

    def clip_loss(self, vlm_features, comp_features, logit_scale):
        """
        CLIP-style contrastive loss for 1 VLM anchor vs N T5 features (1 positive + N-1 negatives).
        Args:
            vlm_features(Tensor): Shape [1, D] - VLM features (anchor)
            comp_features(Tensor): Shape [N, D] - T5 comp features (index 0 is positive, rest are negatives)
            logit_scale: Temperature scaling parameter
        Returns:
            loss(Tensor): Scalar loss value
        """
        # L2 normalize features
        vlm_features = torch.nn.functional.normalize(vlm_features.float(), p=2, dim=1)  # [1, D]
        comp_features = torch.nn.functional.normalize(comp_features.float(), p=2, dim=1)  # [N, D]

        # Compute similarity: [1, D] @ [D, N] = [1, N]
        # vlm_features 与每个 T5 特征的相似度
        logits = vlm_features @ comp_features.T  # [1, N]

        # Apply temperature scaling with clamp to prevent numerical instability
        logit_scale_value = logit_scale.exp().clamp(max=100.0)
        logits = logits * logit_scale_value  # [1, N]

        # Label: VLM should match the first T5 feature (index 0, which is the positive example)
        labels = torch.zeros(1, dtype=torch.long, device=logits.device)  # [0]

        # Cross-entropy loss: VLM 应该选择 index=0 的 T5 特征（正例）
        loss = nn.functional.cross_entropy(logits, labels)

        return loss

    def hungarian_matching(self, vlm_context_emb, t5_context_emb, t5_attention_mask):
        """
        Args:
            vlm_context_emb: Shape (b t1 c)  
            t5_context_emb: Shape (b t2 c)  
            t5_attention_mask: Shape (b t2)  
        """
        valid_t5_len = t5_attention_mask.sum().item()
        t5_feat = t5_context_emb[:,:valid_t5_len].float().squeeze(0)    # (t2, c)
        vlm_feat = vlm_context_emb.float().squeeze(0)                   # (t1, c)

        with torch.no_grad():
            # Compute L1 distance matrix (Manhattan distance): (L, 256)
            # Using torch.cdist with p=1 for L1 distance
            cost_matrix = torch.cdist(t5_feat, vlm_feat, p=1)  # (L, 256)

            # Run Hungarian algorithm (scipy expects numpy array on CPU)
            cost_matrix_np = cost_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

            # Convert indices back to tensor
            row_ind = torch.tensor(row_ind, device=t5_context_emb.device, dtype=torch.long)
            col_ind = torch.tensor(col_ind, device=vlm_context_emb.device, dtype=torch.long)

            matched_t5_feat = t5_context_emb[0][row_ind]  # (min(L, 256), 5120)

        matched_vlm_feat = vlm_context_emb[0][col_ind]

        return matched_t5_feat, matched_vlm_feat

    @staticmethod
    def pad_to_target_len(embed, attn_mask, target_len):
        """Pad embedding and attention mask to target sequence length."""
        valid_len = attn_mask.sum().item()
        embed = embed[:, :valid_len]
        attn_mask = attn_mask[:, :valid_len]
        b, l, c = embed.shape
        if l < target_len:
            pad_len = target_len - l
            embed_pad = torch.zeros(b, pad_len, c, device=embed.device, dtype=embed.dtype)
            embed = torch.cat([embed, embed_pad], dim=1)
            if attn_mask is not None:
                mask_pad = torch.zeros(b, pad_len, device=attn_mask.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([attn_mask, mask_pad], dim=1)
        return embed, attn_mask

    def calc_alignment_losses(
        self, 
        vlm_context_emb, 
        vlm_attention_mask, 
        text_context_emb, 
        text_attention_mask, 
        neg_text_context, 
        neg_text_attention_mask, 
        t5_text_len=1024
    ):
        t5_text_context_before_qformer_vec = text_context_emb[0, :text_attention_mask.sum().item()].mean(dim=0) # (5120,)

        vlm_context_emb_before_qformer_vec = vlm_context_emb[0].mean(dim=0) # (5120,)
        before_qformer_l1_loss = torch.nn.functional.l1_loss(
            vlm_context_emb_before_qformer_vec.float(), 
            t5_text_context_before_qformer_vec.float())

        # ========== Hungarian algorithm matching between anchor(VLM) and positive(text_context_emb)
        matched_t5_feat, matched_vlm_feat = self.hungarian_matching(
            vlm_context_emb=vlm_context_emb, 
            t5_context_emb=text_context_emb, 
            t5_attention_mask=text_attention_mask)
        hungarian_l1_loss = torch.nn.functional.l1_loss(matched_vlm_feat.float(), matched_t5_feat.float())

        # ========== encode the negative context ========== 
        neg_text_context_lens = None
        neg_text_context_emb = self.text_embedding(
            torch.stack([
                torch.cat(
                    # [u, u.new_zeros(max(0, self.text_len - u.size(0)), u.size(1))])
                    [u, u.new_zeros(max(0, t5_text_len - u.size(0)), u.size(1))])
                for u in neg_text_context
            ]))
        neg_text_context_emb_before_qformer_vec = [
            u[:m.sum().item()].mean(dim=0) for u, m in zip(neg_text_context_emb, neg_text_attention_mask)
        ]
        neg_text_context_emb_before_qformer_vec = torch.stack(neg_text_context_emb_before_qformer_vec, dim=0) # (b, c)

        all_text_context_vecs = torch.cat([
            t5_text_context_before_qformer_vec.unsqueeze(0), 
            neg_text_context_emb_before_qformer_vec], dim=0)

        before_qformer_contrastive_loss = self.clip_loss(
            vlm_features=vlm_context_emb_before_qformer_vec.unsqueeze(0).float(),           # [1, D]
            comp_features=all_text_context_vecs.float(),     # [32, D], index 0 is positive
            logit_scale=self.logit_scale
        )

        # ========== Use qformer to align ==========
        t5_context_emb_padded, _ = self.pad_to_target_len(
            text_context_emb, text_attention_mask, 512,
        )
        neg_text_context_emb_padded, _ = self.pad_to_target_len(
            neg_text_context_emb, neg_text_attention_mask, 512,
        )
        vlm_context_emb_padded, _ = self.pad_to_target_len(
            vlm_context_emb, vlm_attention_mask, 512,
        )
        # 正例: align_qformer 处理 T5 特征和 VLM 特征
        align_targ, align_feat = self.align_qformer(
            t5_embed=t5_context_emb_padded,
            t5_attention_mask=None,
            vlm_embed=vlm_context_emb_padded
        )
        align_loss = torch.nn.functional.l1_loss(align_feat.float(), align_targ.float())

        # ========== Use qformer to align (contrastive) ==========
        align_negs, _ = self.align_qformer(
            t5_embed=neg_text_context_emb_padded,
            t5_attention_mask=None,
            vlm_embed=vlm_context_emb_padded
        )
        all_comp_features = torch.cat([align_targ, align_negs], dim=0)  # [32, num_queries, out_dim]
        all_comp_pooled = all_comp_features.mean(dim=1)  # [32, out_dim]
        vlm_pooled = align_feat.mean(dim=1)  # [1, out_dim]
        contrastive_loss_val = self.clip_loss(
            vlm_features=vlm_pooled,           # [1, D]
            comp_features=all_comp_pooled,     # [32, D], index 0 is positive
            logit_scale=self.logit_scale
        )

        loss_dict = dict(
            align=align_loss,
            contrastive=contrastive_loss_val,
            before_qformer_l1_loss=before_qformer_l1_loss,
            before_qformer_contrastive_loss=before_qformer_contrastive_loss,
            hungarian_l1_loss=hungarian_l1_loss,
        )

        return loss_dict

    def calc_repa_losses(
        self, 
        align_source, 
        align_target, 
        grid_sizes, 
        **kwargs, 
    ):
        
        f, h, w = align_target.shape[2:]
        target = rearrange(align_target.float(), "b c f h w -> b f (h w) c")
        
        trd_losses = 0
        for feat in align_source:
            source = rearrange(feat.float(), "b (f h w) c -> b c f h w", 
                               f=grid_sizes[1], h=grid_sizes[2]//2, w=grid_sizes[3]//2)
            source = torch.nn.functional.interpolate(
                source[:,:,1:].float(), 
                (f, h, w), mode="trilinear"
            )
            source = rearrange(source, "b c f h w -> b f (h w) c")

            trd_loss, _, _ = self.trd_loss_fn(
                rearrange(source, "b f (h w) c -> b c f h w", h=h, w=w), 
                rearrange(target, "b f (h w) c -> b c f h w", h=h, w=w)
            )
            trd_losses = trd_losses + trd_loss

        trd_losses = trd_losses / len(align_source)
        return trd_losses

    def gen_rope_embeddings(self, ref_id, 
                            ff_context_emb, ff_grid_sizes, 
                            x_emb, grid_sizes, 
                            lf_context_emb, lf_grid_sizes, 
                            ref_context_emb, ref_grid_sizes, 
                            video_context_emb, video_grid_sizes, 
                            f_shift: int = 5, ):
        """
        """
        # --- compute the shifted rotary embedding for noise latents and context latents ---
        shift_f_size = 0
        shift_h_size = 0
        shift_w_size = 0

        # (optional) get the rotary embedding of ff_context_emb
        if ff_context_emb is not None:
            freqs_ff_context = shift_rope(ff_context_emb, ff_grid_sizes, 
                                        self.freqs, self.dim // self.num_heads, 
                                        shift_f=False, 
                                        shift_h=False, 
                                        shift_w=False, 
                                        shift_f_size=shift_f_size, 
                                        shift_h_size=shift_h_size, 
                                        shift_w_size=shift_w_size)
            shift_f_size += int(ff_grid_sizes[0][0].item())

        # get the rotary embedding of x
        freqs_x = shift_rope(x_emb, grid_sizes, 
                             self.freqs, self.dim // self.num_heads, 
                             shift_f=True, 
                             shift_h=False, 
                             shift_w=False, 
                             shift_f_size=shift_f_size, 
                             shift_h_size=shift_h_size, 
                             shift_w_size=shift_w_size)
        shift_f_size += int(grid_sizes[0][0].item())
        shift_h_size += 0
        shift_w_size += 0

        # (optional) get the rotary embedding of lf_context_emb
        if lf_context_emb is not None:
            freqs_lf_context = shift_rope(lf_context_emb, lf_grid_sizes, 
                                          self.freqs, self.dim // self.num_heads, 
                                          shift_f=True, 
                                          shift_h=False, 
                                          shift_w=False, 
                                          shift_f_size=shift_f_size, 
                                          shift_h_size=shift_h_size, 
                                          shift_w_size=shift_w_size)
            shift_f_size += 1

        # (optional) get the rotary embedding of ref_context_emb
        if ref_context_emb is not None:
            freqs_ref_context, shift_f_size, shift_h_size, shift_w_size = self._gen_rope_default4(
                ref_context_emb, ref_grid_sizes, ref_id, 
                shift_f_size, shift_h_size, shift_w_size, 
                f_shift=f_shift
            )

        # (optional) get the rotary embedding of video_context_emb
        if video_context_emb is not None:
            freqs_video_context = shift_rope(video_context_emb, 
                                             video_grid_sizes, 
                                             self.freqs, self.dim // self.num_heads, 
                                             shift_f=True, 
                                             shift_h=False, 
                                             shift_w=False, 
                                             shift_f_size=shift_f_size, 
                                             shift_h_size=shift_h_size, 
                                             shift_w_size=shift_w_size)

        freqs = []
        if ff_context_emb is not None:
            freqs.append(freqs_ff_context)      # append f_freqs
        freqs.append(freqs_x)                   # append x_freqs
        if lf_context_emb is not None:
            freqs.append(freqs_lf_context)      # append lf_freqs
        if ref_context_emb is not None:
            freqs.append(freqs_ref_context)     # append ref_freqs
        if video_context_emb is not None:
            freqs.append(freqs_video_context)   # append vid_freqs
        freqs = torch.cat(freqs, dim=0)

        return freqs

    def _gen_rope_default4(self, ref_context_emb, ref_grid_sizes, ref_id, 
                          shift_f_size, shift_h_size, shift_w_size, f_shift):
        """
        默认方案：基于类别的硬编码偏移
        """
                       # rope_shift gap = 5
        freqs_ref_context = []
        for idx, grids in enumerate(ref_grid_sizes):
            _ref_id = ref_id.view(-1)
            shift_f_size += f_shift     # += 3
            if _ref_id[idx] < 100:      # human
                shift_h_size = 0
                shift_w_size = 0
            elif _ref_id[idx] < 200:    # object
                if idx != 0 and _ref_id[idx-1] < 100:
                    pass
                    # shift_h_size = grids[1].item()  # shift_h_size = h
                    # shift_w_size = grids[2].item()  # shift_w_size = w
            elif _ref_id[idx] < 300:    # scene
                if idx != 0 and _ref_id[idx-1] < 200:
                    pass
                    # shift_h_size = grids[1].item() + grids[1].item()  # shift_h_size = h + h
                    # shift_w_size = grids[2].item() + grids[2].item()  # shift_w_size = w + w
            elif _ref_id[idx] >= 300:   # memory
                if idx != 0 and _ref_id[idx-1] < 300:
                    shift_f_size = 100
                    shift_h_size = 0
                    shift_w_size = 0
            
            freqs_ref = shift_rope(ref_context_emb, 
                                   grids.unsqueeze(0), 
                                   self.freqs, self.dim // self.num_heads, 
                                   shift_f=True, 
                                   shift_h=True, 
                                   shift_w=True, 
                                   shift_f_size=shift_f_size, 
                                   shift_h_size=shift_h_size, 
                                   shift_w_size=shift_w_size)
            freqs_ref_context.append(freqs_ref)
            shift_f_size += int(grids[0].item())

        return torch.cat(freqs_ref_context, dim=0), shift_f_size, shift_h_size, shift_w_size

def focal_heatmap_loss(pred, gt, alpha=2, beta=4):
    """
    pred: 预测热图 (batch, c, h, w), 经过 sigmoid 归一化到 0-1
    gt:   真值热图 (batch, c, h, w), 高斯分布生成，峰值为 1
    """
    # 1. 压制数值稳定性错误
    pos_inds = gt.ge(0.8)  # 真正的高峰点 (center)
    neg_inds = gt.lt(0.8)  # 所有的非中心点 (包括高斯晕和背景)

    # 2. 负样本权重：离中心越近 (gt值越大)，该点的负样本惩罚越小
    # 这允许网络在中心附近有模糊的预测，而不是强制非中心点立刻变为0
    neg_weights = torch.pow(1 - gt, beta)

    loss = 0

    # 3. 正样本 Loss (log 惩罚)
    # 如果 pred 很低，(1-pred)^alpha 会很大，导致 loss 很大 -> 强迫拉高响应
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds.float()

    # 4. 负样本 Loss (背景)
    # 如果 pred 很高，pred^alpha 会很大 -> 强迫压低背景
    # 结合 neg_weights，离中心越远的点，压制力度越大
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds.float()

    # 5. 归一化
    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss


def weighted_mse_loss(pred, gt, weight=100):
    # weight: 用于放大前景误差的系数
    mse = (pred - gt) ** 2
    
    # 创建权重图：只要 GT 有值的地方，权重就变大
    # 这里假设 GT 是高斯图，数值 > 0
    weights = torch.ones_like(gt)
    weights[gt > 0.1] = weight  # 对有响应区域增加 100 倍惩罚
    
    return (mse * weights).mean()


class TRDLoss(nn.Module):
    def __init__(self, margin=0.1, lambda_temporal=1.0):
        """
        TRD损失函数实现
        Args:
            margin: 损失边际值，过滤微小差异（论文中VideoMAEv2对应0.1）
            lambda_temporal: 时间损失分量权重（论文中与空间分量等权，设为1.0）
        """
        super().__init__()
        self.margin = margin
        self.lambda_temporal = lambda_temporal

    def _compute_pairwise_similarity(self, features):
        """
        计算token对之间的余弦相似度矩阵
        Args:
            features: 输入特征 [B, T, N, D]，其中B=批次，T=时间帧数，N=空间token数，D=特征维度
        Returns:
            similarity: 相似度矩阵 [B, T, N, N]
        """
        B, T, N, D = features.shape
        
        # 特征归一化（L2归一化，确保余弦相似度计算正确）
        features_norm = torch.nn.functional.normalize(features, p=2, dim=-1)
        
        # 计算每个时间帧内的token对相似度：[B,T,N,D] @ [B,T,D,N] = [B,T,N,N]
        similarity = torch.matmul(features_norm, features_norm.transpose(-1, -2))
        return similarity

    def _align_dimensions(self, vdm_features, vfm_features):
        """
        对齐VDM和VFM的特征维度（时空维度插值匹配）
        Args:
            vdm_features: VDM隐藏特征 [B, T_vdm, N_vdm, D]
            vfm_features: VFM特征 [B, T_vfm, N_vfm, D]
        Returns:
            vdm_features_aligned: 对齐后的VDM特征 [B, T_vfm, N_vfm, D]
        """
        B, T_vdm, N_vdm, D = vdm_features.shape
        _, T_vfm, N_vfm, _ = vfm_features.shape
        
        # 1. 时间维度对齐：插值到VFM的时间帧数
        # 重塑为[B*T_vdm, N_vdm, D]用于插值
        vdm_features_reshaped = vdm_features.reshape(B*T_vdm, N_vdm, D)
        # 时间维度插值：[B*T_vdm, N_vdm, D] → [B*T_vfm, N_vdm, D]
        vdm_features_time_aligned = torch.nn.functional.interpolate(
            vdm_features_reshaped.transpose(1, 2),  # [B*T_vdm, D, N_vdm]
            size=T_vfm, 
            mode='linear', 
            align_corners=False
        ).transpose(1, 2)  # [B*T_vfm, N_vdm, D]
        vdm_features_time_aligned = vdm_features_time_aligned.reshape(B, T_vfm, N_vdm, D)
        
        # 2. 空间维度对齐：插值到VFM的空间token数
        vdm_features_aligned = []
        for b in range(B):
            frame_features = []
            for t in range(T_vfm):
                # 单帧特征：[N_vdm, D] → [N_vfm, D]
                frame_feat = F.interpolate(
                    vdm_features_time_aligned[b, t].unsqueeze(0).unsqueeze(0),  # [1,1,N_vdm,D]
                    size=(N_vfm, D),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)  # [N_vfm, D]
                frame_features.append(frame_feat)
            vdm_features_aligned.append(torch.stack(frame_features, dim=0))  # [T_vfm, N_vfm, D]
        
        return torch.stack(vdm_features_aligned, dim=0)  # [B, T_vfm, N_vfm, D]

    def forward_dump(self, vdm_hidden_states, vfm_features):
        """
        前向传播计算TRD损失
        Args:
            vdm_hidden_states: VDM去噪Transformer隐藏状态 [B, T_vdm, N_vdm, D_vdm]
            vfm_features: VFM编码特征 [B, T_vfm, N_vfm, D_vfm]（已归一化）
        Returns:
            total_loss: TRD总损失（空间损失 + 时间损失）
            spatial_loss: 空间关系损失分量
            temporal_loss: 时间关系损失分量
        """
        # 1. VDM特征投影与维度统一
        B = vdm_hidden_states.shape[0]
        
        # 3. 计算空间相似度矩阵（帧内token对关系）
        vdm_spatial_sim = self._compute_pairwise_similarity(vdm_hidden_states)  # [B, T_vfm, N_vfm, N_vfm]
        vfm_spatial_sim = self._compute_pairwise_similarity(vfm_features)  # [B, T_vfm, N_vfm, N_vfm]
        
        # 4. 计算空间损失（L1距离 + margin过滤）
        spatial_diff = torch.abs(vdm_spatial_sim - vfm_spatial_sim)
        spatial_diff = torch.maximum(spatial_diff - self.margin, torch.tensor(0.0, device=spatial_diff.device))
        T_vfm, N_vfm = vfm_spatial_sim.shape[1], vfm_spatial_sim.shape[2]
        spatial_loss = spatial_diff.sum() / (B * T_vfm * N_vfm * N_vfm)  # 平均到每个token对
        
        # 5. 计算时间相似度矩阵（跨帧token对关系）
        temporal_loss = 0.0
        for b in range(B):
            for t in range(T_vfm):
                # 当前帧token特征 [N_vfm, D_vfm]
                current_frame_feat = vdm_hidden_states[b, t]
                # 其他所有帧 [T_vfm-1, N_vfm, D_vfm]
                other_frames_feat = vdm_hidden_states[b, [i for i in range(T_vfm) if i != t]]
                
                # 当前帧与其他帧的token相似度：[N_vfm, D] @ [T-1, D, N] = [T-1, N_vfm, N_vfm]
                current_norm = torch.nn.functional.normalize(current_frame_feat, p=2, dim=-1)
                other_norm = torch.nn.functional.normalize(other_frames_feat, p=2, dim=-1)
                vdm_temp_sim = torch.matmul(current_norm, other_norm.transpose(-1, -2))  # [T-1, N_vfm, N_vfm]
                
                # VFM的时间相似度计算
                vfm_current_norm = torch.nn.functional.normalize(vfm_features[b, t], p=2, dim=-1)
                vfm_other_norm = torch.nn.functional.normalize(vfm_features[b, [i for i in range(T_vfm) if i != t]], p=2, dim=-1)
                vfm_temp_sim = torch.matmul(vfm_current_norm, vfm_other_norm.transpose(-1, -2))  # [T-1, N_vfm, N_vfm]
                
                # 时间损失计算（L1距离 + margin过滤）
                temp_diff = torch.abs(vdm_temp_sim - vfm_temp_sim)
                temp_diff = torch.maximum(temp_diff - self.margin, torch.tensor(0.0, device=temp_diff.device))
                temporal_loss += temp_diff.sum()
        
        # 时间损失归一化
        temporal_loss = temporal_loss / (B * T_vfm * (T_vfm - 1) * N_vfm * N_vfm)
        temporal_loss *= self.lambda_temporal
        
        # 6. 总损失
        total_loss = spatial_loss + temporal_loss
        
        return total_loss, spatial_loss, temporal_loss

    def forward(self, x_vdm, target_vfm):
        """
        x_vdm: [B, C1, F_vdm, H/16, W/16]  -> 维度 [(F+3)//4, H // 16, W // 16]
        target_vfm: [B, C2, F_vfm, H/16, W/16] -> 维度 [F//2, H // 3 // 16, W // 3 // 16], VJEPA2
        """
        B, C1, F_vdm, H, W = x_vdm.shape
        _, C2, F_vfm, _, _ = target_vfm.shape

        # 1. 时间轴对齐 (Temporal Alignment)
        # 使用 trilinear 插值将 VDM 的特征平滑缩放到 VFM 的时间长度
        # align_corners=False 通常对深度学习特征更友好
        x_vdm_resized = torch.nn.functional.interpolate(
            x_vdm,
            size=(F_vfm, H, W),
            mode='trilinear',
            align_corners=False
        )

        # 2. 空间维度转置以便进行 Channel Projection
        # [B, C1, F, H, W] -> [B, F, H, W, C1]
        x_vdm_resized = x_vdm_resized.permute(0, 2, 3, 4, 1)

        # 3. 特征投影 (Feature Projection)
        # 将 VDM 的特征维度映射到 VFM 维度
        B, F, H_proj, W_proj, align_dim = x_vdm_resized.shape

        # 4. 计算 REPA Loss (Cosine Similarity)
        # 将 target_vfm 转换为相同形状 [B, F, H, W, C2]
        target_vfm = target_vfm.permute(0, 2, 3, 4, 1)

        loss_type = "token_relation_distillation"
        if loss_type == 'token_relation_distillation':
            align = x_vdm_resized.flatten(2, 3)  # B, F, H*W, C
            align_target = target_vfm.flatten(2, 3)  # B, F, H*W, C

            align = torch.nn.functional.normalize(align, dim=-1)
            align_target = torch.nn.functional.normalize(align_target, dim=-1)
            assert align.shape[-1] == align_target.shape[-1], "The last dimension of align and align_target must be the same"

            F = align.shape[1]
            align_sim = torch.bmm(align.flatten(0, 1), align.flatten(1, 2).unsqueeze(1).expand(-1, F, -1, -1).flatten(0, 1).transpose(1, 2))
            align_target_sim = torch.bmm(align_target.flatten(0, 1), align_target.flatten(1, 2).unsqueeze(1).expand(-1, F, -1, -1).flatten(0, 1).transpose(1, 2))
            assert align_sim.shape == align_target_sim.shape

            loss = nn.functional.relu((align_sim - align_target_sim).abs() - self.margin).mean()
        elif self.loss_type == 'cosine_similarity':
            pass
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        return loss, None, None

def batch_cosine_similarity(a):
    """
    使用 F.cosine_similarity 计算 [b, t, c] 元素间的相似度
    输出尺寸: [b, t, t]
    """
    normalized_a = torch.nn.functional.normalize(a.float(), p=2, dim=-1)

    sim_matrix = torch.matmul(normalized_a, normalized_a.permute(0,2,1))
    
    return sim_matrix

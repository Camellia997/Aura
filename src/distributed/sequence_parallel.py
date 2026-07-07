# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp
from einops import rearrange

# from src.modules.hyvideo_edit_v1 import sinusoidal_embedding_1d
from .ulysses import distributed_attention
from .util import gather_forward, get_rank, get_world_size


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


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_world_size()
        sp_rank = get_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


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
    # s, c = x.size(1), attn_dim // 2
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


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    use_gradient_checkpointing=False,
    **infer_kwargs, 
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    if isinstance(context, dict) and "ref_context" in context:
        ref_context = context["ref_context"]
        x = [torch.cat([u, v], dim=1) for u, v in zip(x, ref_context)]
    else:
        ref_context = None

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    seq_len = seq_lens.max().item()
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    text_context = context["text_context"]
    context_lens = None
    text_context = self.text_embedding(
        torch.stack([
            torch.cat(
                # [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                [u, u.new_zeros(1024 - u.size(0), u.size(1))])
            for u in text_context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=text_context,
        context_lens=context_lens)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)

    if ref_context is not None:
        x = [u[:, :-v.shape[1]] for u, v in zip(x, ref_context)]

    return [u.float() for u in x]


def sp_attn_forward(
    self, 
    x, 
    seq_lens, 
    grid_sizes, 
    freqs, 
    dtype=torch.bfloat16, 
    **kwargs
):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

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

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


def sp_dit_forward_edit_v1(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    use_gradient_checkpointing=False,
    **infer_kwargs, 
):
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

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

    # # reference context
    # if isinstance(context, dict) and "ref_context" in context:
    #     ref_context = context["ref_context"]    # (b*n c 1 h w)
    #     ref_seq_lens = ref_context.size(0) * ref_context.size(3) * ref_context.size(4) // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
    #     num_valid_context += ref_context.size(0)
    # else:
    #     ref_context = None
    #     ref_seq_lens = 0
    ref_context = None
    ref_seq_lens = 0

    # video context
    if isinstance(context, dict) and "video_context" in context:
        video_context = context["video_context"]    # (b c n h w)
        video_seq_lens = video_context.size(2) * video_context.size(3) * video_context.size(4) // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        # x = [torch.cat([u, v], dim=1) for u, v in zip(x, video_context)]
    else:
        video_context = None
        video_seq_lens = 0

    # # (Optional) memory token
    # if ref_context is not None:
    #     h_, w_ = x[0].shape[-2:]
    #     num_memory_context = 6 - num_valid_context
    #     memory_context = self.memory_tokens.repeat(h_*w_*num_memory_context, 1).unsqueeze(0) # (b f c)
    #     memory_context = rearrange(memory_context, "b (f h w) c -> (b f) c 1 h w", f=num_memory_context, h=h_, w=w_)
    #     ref_context = torch.concat([ref_context, memory_context], dim=0)
    #     ref_seq_lens += num_memory_context * h_ * w_ // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
    #     assert "ref_id" in context
    #     ref_id = context["ref_id"].view(1,-1)
    #     ref_id = torch.cat([ref_id, torch.tensor([300] * num_memory_context).to(ref_id).view(1, -1)], dim=1)
    # else:
    #     ref_id = context["ref_id"].view(1,-1)
    ref_id = context["ref_id"].view(1,-1)

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
                                     f_shift=5, )
    
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

    # --- encode context ---
    # text context
    text_context_lens = None
    text_context = context["text_context"]
    text_attention_mask = context["text_attention_mask"]
    text_context_emb = self.text_embedding(
        torch.stack([
            torch.cat(
                # [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                [u, u.new_zeros(max(0, 1024 - u.size(0)), u.size(1))])
            for u in text_context
        ]))

    # 推理的时候，默认vlm_context_emb只做一次，在dit外面完成
    vlm_context_emb = context["vlm_context_emb"]

    # merge vlm_embedding with t5_embedding
    vlm_context_emb_proj = self.vlm_connector_proj(vlm_context_emb)

    # # [DEBUG] t5和vlm特征concat后注入
    # vlm_context_emb_proj = self.merge_vlm_embedding(
    #     text_context_emb=text_context_emb, 
    #     vlm_context_emb=vlm_context_emb_proj, 
    #     text_attention_mask=text_attention_mask
    # )

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

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    freqs = torch.chunk(freqs, get_world_size(), dim=0)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs if self.apply_rope_in_selfattn else freqs, 
        context=vlm_context_emb_proj,
        # context=text_context_emb,
        context_lens=256, 
        # vlm_context=vlm_context_emb_proj,   # [DEBUG]
        )

    self.face_masks = {}
    for block_id, block in enumerate(self.blocks):
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

     # unpatchify
    x = self.unpatchify(x, grid_sizes)
        
    if ff_context is not None:
        x = [u[:, v.shape[1]:] for u, v in zip(x, ff_context)]

    x = [u[:, :v[1]] for u, v in zip(x, x_shape)]

    return [u.float() for u in x]


def sp_dit_forward_edit_v1_high(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    use_gradient_checkpointing=False,
    **infer_kwargs, 
):
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

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
        # print(f"HIGH_NOISE_MODEL: {ref_seq_lens}")
        num_valid_context += ref_context.size(0)
    else:
        ref_context = None
        ref_seq_lens = 0

    # video context
    if isinstance(context, dict) and "video_context" in context:
        video_context = context["video_context"]    # (b c n h w)
        video_seq_lens = video_context.size(2) * video_context.size(3) * video_context.size(4) // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        # x = [torch.cat([u, v], dim=1) for u, v in zip(x, video_context)]
    else:
        video_context = None
        video_seq_lens = 0

    # (Optional) memory token
    if ref_context is not None:
        h_, w_ = x[0].shape[-2:]
        num_memory_context = 6 - num_valid_context
        memory_context = self.memory_tokens.repeat(h_*w_*num_memory_context, 1).unsqueeze(0) # (b f c)
        memory_context = rearrange(memory_context, "b (f h w) c -> (b f) c 1 h w", f=num_memory_context, h=h_, w=w_)
        ref_context = torch.concat([ref_context, memory_context], dim=0)
        ref_seq_lens += num_memory_context * h_ * w_ // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        assert "ref_id" in context
        ref_id = context["ref_id"].view(1,-1)
        ref_id = torch.cat([ref_id, torch.tensor([300] * num_memory_context).to(ref_id).view(1, -1)], dim=1)
    else:
        ref_id = context["ref_id"].view(1,-1)

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
                                     f_shift=5, )
    
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

    # --- encode context ---
    # text context
    text_context_lens = None
    text_context = context["text_context"]
    text_attention_mask = context["text_attention_mask"]
    text_context_emb = self.text_embedding(
        torch.stack([
            torch.cat(
                # [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                [u, u.new_zeros(max(0, 1024 - u.size(0)), u.size(1))])
            for u in text_context
        ]))

    # 推理的时候，默认vlm_context_emb只做一次，在dit外面完成
    vlm_context_emb = context["vlm_context_emb"]

    # merge vlm_embedding with t5_embedding
    vlm_context_emb_proj = self.vlm_connector_proj(vlm_context_emb)

    # # [DEBUG] t5和vlm特征concat后注入
    # vlm_context_emb_proj = self.merge_vlm_embedding(
    #     text_context_emb=text_context_emb, 
    #     vlm_context_emb=vlm_context_emb_proj, 
    #     text_attention_mask=text_attention_mask
    # )

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

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    freqs = torch.chunk(freqs, get_world_size(), dim=0)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs if self.apply_rope_in_selfattn else freqs, 
        # context=vlm_context_emb_proj,
        context=text_context_emb,
        context_lens=256, 
        vlm_context=vlm_context_emb_proj,   # [DEBUG]
        )

    self.face_masks = {}
    for block_id, block in enumerate(self.blocks):
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

     # unpatchify
    x = self.unpatchify(x, grid_sizes)
        
    if ff_context is not None:
        x = [u[:, v.shape[1]:] for u, v in zip(x, ff_context)]

    x = [u[:, :v[1]] for u, v in zip(x, x_shape)]

    return [u.float() for u in x]


def sp_dit_forward_edit_v1_low(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    use_gradient_checkpointing=False,
    **infer_kwargs, 
):
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

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
        # print(f" ----> LOW_NOISE_MODEL {ref_seq_lens}")
        num_valid_context += ref_context.size(0)
    else:
        ref_context = None
        ref_seq_lens = 0

    # video context
    if isinstance(context, dict) and "video_context" in context:
        video_context = context["video_context"]    # (b c n h w)
        video_seq_lens = video_context.size(2) * video_context.size(3) * video_context.size(4) // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        # x = [torch.cat([u, v], dim=1) for u, v in zip(x, video_context)]
    else:
        video_context = None
        video_seq_lens = 0

    # (Optional) memory token
    if ref_context is not None:
        h_, w_ = x[0].shape[-2:]
        num_memory_context = 6 - num_valid_context
        memory_context = self.memory_tokens.repeat(h_*w_*num_memory_context, 1).unsqueeze(0) # (b f c)
        memory_context = rearrange(memory_context, "b (f h w) c -> (b f) c 1 h w", f=num_memory_context, h=h_, w=w_)
        ref_context = torch.concat([ref_context, memory_context], dim=0)
        ref_seq_lens += num_memory_context * h_ * w_ // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        assert "ref_id" in context
        ref_id = context["ref_id"].view(1,-1)
        ref_id = torch.cat([ref_id, torch.tensor([300] * num_memory_context).to(ref_id).view(1, -1)], dim=1)
    else:
        ref_id = context["ref_id"].view(1,-1)

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
                                     f_shift=5, )
    
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

    # --- encode context ---
    # text context
    text_context_lens = None
    text_context = context["text_context"]
    text_attention_mask = context["text_attention_mask"]
    text_context_emb = self.text_embedding(
        torch.stack([
            torch.cat(
                # [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                [u, u.new_zeros(max(0, 1024 - u.size(0)), u.size(1))])
            for u in text_context
        ]))

    # 推理的时候，默认vlm_context_emb只做一次，在dit外面完成
    vlm_context_emb = context["vlm_context_emb"]

    # merge vlm_embedding with t5_embedding
    vlm_context_emb_proj = self.vlm_connector_proj(vlm_context_emb)

    # # [DEBUG] t5和vlm特征concat后注入
    # vlm_context_emb_proj = self.merge_vlm_embedding(
    #     text_context_emb=text_context_emb, 
    #     vlm_context_emb=vlm_context_emb_proj, 
    #     text_attention_mask=text_attention_mask
    # )

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

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    freqs = torch.chunk(freqs, get_world_size(), dim=0)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs if self.apply_rope_in_selfattn else freqs, 
        # context=vlm_context_emb_proj,
        context=text_context_emb,
        context_lens=256, 
        vlm_context=vlm_context_emb_proj,   # [DEBUG]
        )

    self.face_masks = {}
    for block_id, block in enumerate(self.blocks):
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

     # unpatchify
    x = self.unpatchify(x, grid_sizes)
        
    if ff_context is not None:
        x = [u[:, v.shape[1]:] for u, v in zip(x, ff_context)]

    x = [u[:, :v[1]] for u, v in zip(x, x_shape)]

    return [u.float() for u in x]


def sp_dit_forward_edit_v3_ct(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    use_gradient_checkpointing=False,
    **infer_kwargs, 
):
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

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
        # x = [torch.cat([u, v], dim=1) for u, v in zip(x, video_context)]
    else:
        video_context = None
        video_seq_lens = 0

    # (Optional) memory token
    if ref_context is not None:
        h_, w_ = x[0].shape[-2:]
        num_memory_context = 6 - num_valid_context
        # num_memory_context = 3  # 必须有3张memory token图
        memory_context = self.memory_tokens.repeat(h_*w_*num_memory_context, 1).unsqueeze(0) # (b f c)
        memory_context = rearrange(memory_context, "b (f h w) c -> (b f) c 1 h w", f=num_memory_context, h=h_, w=w_)
        ref_context = torch.concat([ref_context, memory_context], dim=0)
        ref_seq_lens += num_memory_context * h_ * w_ // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        assert "ref_id" in context
        ref_id = context["ref_id"].view(1,-1)
        ref_id = torch.cat([ref_id, torch.tensor([300] * num_memory_context).to(ref_id).view(1, -1)], dim=1)
    else:
        ref_id = context["ref_id"].view(1,-1)
    # ref_id = context["ref_id"].view(1,-1)

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
                                     f_shift=3)
    
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

    # --- encode context ---
    # text context
    text_context_lens = None
    text_context = context["text_context"]
    text_attention_mask = context["text_attention_mask"]
    text_context_emb = self.text_embedding(
        torch.stack([
            torch.cat(
                # [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                [u, u.new_zeros(max(0, 1024 - u.size(0)), u.size(1))])
            for u in text_context
        ]))

    # 推理的时候，默认vlm_context_emb只做一次，在dit外面完成
    vlm_context_emb = context["vlm_context_emb"]

    # merge vlm_embedding with t5_embedding
    vlm_context_emb_proj = self.vlm_connector_proj(vlm_context_emb)

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

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    freqs = torch.chunk(freqs, get_world_size(), dim=0)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs if self.apply_rope_in_selfattn else freqs,
        context=text_context_emb,
        context_lens=text_context_lens,
        vlm_context_emb=vlm_context_emb_proj,
        pladis_scale=infer_kwargs.get("pladis_scale", 0.0),
        noise_range=[ff_seq_lens, seq_lens.max().item() - (lf_seq_lens + ref_seq_lens + video_seq_lens)],
        sp_size=get_world_size(),
        sp_rank=get_rank(),
        )

    self.face_masks = {}
    for block_id, block in enumerate(self.blocks):
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)
    # for block_id in range(len(self.blocks)):
    #     self.face_masks[f"block_{block_id}"] = gather_forward(self.face_masks[f"block_{block_id}"].contiguous(), dim=1)

     # unpatchify
    x = self.unpatchify(x, grid_sizes)
        
    if ff_context is not None:
        x = [u[:, v.shape[1]:] for u, v in zip(x, ff_context)]

    x = [u[:, :v[1]] for u, v in zip(x, x_shape)]

    return [u.float() for u in x]


def sp_attn_forward_edit(
    self, 
    x, 
    seq_lens, 
    grid_sizes, 
    freqs, 
    dtype=torch.bfloat16, 
    **kwargs, 
):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

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

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    if kwargs.get("pag", False):
        x = attention_with_identity(
            q, k, v, seq_lens=seq_lens, 
            alpha=1.0
        )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


def sp_hier_fusion_forward(
    self, x, orig_grid_sizes, 
):
    n, h, w = len(orig_grid_sizes), orig_grid_sizes[0,1].item(), orig_grid_sizes[0,2].item()

    # Context Parallel
    x = gather_forward(x, dim=1)

    # split
    noise_token, context_token = x[:, :-(n*h*w)], x[:, -(n*h*w):]

    intra_embed = rearrange(context_token, "b (n f h w) c -> (b n) c f h w", n=n, f=1, h=h, w=w)
    grid_sizes = torch.stack([torch.tensor(x.shape[1:]) for x in intra_embed])
    intra_embed = rearrange(intra_embed, "(b n) c f h w -> (b n) (f h w) c", n=n)
    # intra-subjects interaction
    intra_embed = self.intra_attn(
        self.norm_intra(intra_embed), grid_sizes, self.freqs
    )
    # inter-subjects interaction
    inter_embed = rearrange(intra_embed, "(b n) (f h w) c -> b c (f n) h w", b=1, f=1, h=h, w=w)
    grid_sizes = torch.stack([torch.tensor(x.shape[1:]) for x in inter_embed])
    inter_embed = rearrange(inter_embed, "b c (f n) h w -> b (f n h w) c", n=n, f=1, h=h, w=w)
    inter_embed = self.inter_attn(
        self.norm_inter(inter_embed), grid_sizes, self.freqs
    )
    out_embed = self.ffn(self.norm_out(inter_embed))

    x = torch.cat([noise_token, out_embed], dim=1)

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]

    return x


class EnergyBasedScaling(torch.nn.Module):
    def __init__(self, d_k: int, gamma_min: float = 1.0, gamma_max: float = 1.5, beta: float = 0.1, mu: float = 0.0):
        """
        基于能量的注意力缩放调制（Energy-based Scaling）
        Args:
            d_k: Q/K 的维度（如 768）
            gamma_min: 最小缩放系数（默认 1.0）
            gamma_max: 最大缩放系数（默认 1.5）
            beta: sigmoid 陡峭系数（控制映射灵敏度，默认 0.1）
            mu: logits 平均值的经验阈值（默认 0.0）
        """
        super().__init__()
        self.d_k = d_k
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.beta = beta
        self.mu = mu

    def compute_logits_spread(self, Q: torch.Tensor, K: torch.Tensor) -> float:
        """
        计算注意力 logits 的扩散程度（z_avg）
        Args:
            Q: query 张量，shape = [batch_size, n_q, d_k]
            K: key 张量，shape = [batch_size, n_k, d_k]
        Returns:
            z_avg: 所有 logits 的平均值（标量）
        """
        batch_size, n_q, _ = Q.shape
        _, n_k, _ = K.shape
        
        # 计算 Q@K^T / sqrt(d_k)，shape = [batch_size, n_q, n_k]
        logits = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=Q.dtype))
        
        # 计算全局平均值（跨 batch、n_q、n_k）
        z_avg = logits.mean().item()  # 转为标量，避免批量内差异干扰
        return z_avg

    def forward(self, Q: torch.Tensor, K: torch.Tensor, scale_target: str = "key") -> tuple[torch.Tensor, torch.Tensor]:
        """
        应用自适应缩放
        Args:
            Q: query 张量，shape = [batch_size, n_q, d_k]
            K: key 张量，shape = [batch_size, n_k, d_k]
            scale_target: 缩放目标（"key" 或 "query"，默认 "key"）
        Returns:
            Q': 调制后的 query（若 scale_target="query"）
            K': 调制后的 key（若 scale_target="key"）
        """
        # 步骤1：计算 logits 扩散程度 z_avg
        z_avg = self.compute_logits_spread(Q, K)
        
        # 步骤2：通过单调函数 f(·) 计算 γ_e
        # f(z_avg) = gamma_min + (gamma_max - gamma_min) * sigmoid(-beta * (z_avg - mu))
        sigmoid_term = torch.sigmoid(torch.tensor(-self.beta * (z_avg - self.mu), dtype=Q.dtype))
        gamma_e = self.gamma_min + (self.gamma_max - self.gamma_min) * sigmoid_term
        
        # 步骤3：应用缩放（仅缩放 key 或 query）
        if scale_target == "key":
            K_scaled = K * gamma_e
            return Q, K_scaled
        elif scale_target == "query":
            Q_scaled = Q * gamma_e
            return Q_scaled, K
        else:
            raise ValueError(f"scale_target 必须是 'key' 或 'query'，当前为 {scale_target}")


def attention_with_identity(q, k, v, seq_lens, alpha):
    # q,k,v: [B, L, H, D]
    b, l, h, d = q.shape

    q = q.transpose(1, 2)  # [B, H, L, D]
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    # [B, H, L, L]

    # Identity perturbation（不显式构造矩阵）
    eye = torch.eye(l, device=q.device, dtype=q.dtype)
    # attn_logits = attn_logits + alpha * eye.unsqueeze(0).unsqueeze(0)

    # attn = torch.softmax(attn_logits, dim=-1)
    # out = torch.matmul(attn, v)  # [B, H, L, D]
    out = torch.matmul(eye.unsqueeze(0).unsqueeze(0), v)  # [B, H, L, D]

    return out.transpose(1, 2)   # [B, L, H, D]
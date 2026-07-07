# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
sys.path.append(os.getcwd())
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
from PIL import Image
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
import cv2

from src.distributed.fsdp import shard_model
from src import (
    WanVAE, 
    T5EncoderModel, 
)
from src.pipelines.pipeline_utils import build_dit_models
from src.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas, 
    retrieve_timesteps
)
from src.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.distributed.sequence_parallel import (
    sp_attn_forward, 
    sp_dit_forward, 
    sp_attn_forward_edit, 
    sp_dit_forward_edit_v1, 
    sp_dit_forward_edit_v1_high, 
    sp_dit_forward_edit_v1_low, 
    sp_dit_forward_edit_v3_ct, 
)
from src.distributed.util import get_world_size
from safetensors.torch import load_file

T5_TEXT_LEN = 1024

def load_state_dict(model_dir, postfix=".safetensors"):
    chunk_path_list = [os.path.join(model_dir, name) for name in os.listdir(model_dir) if name.endswith(postfix)]
    chunk_length = len(chunk_path_list)

    state_dict = {}
    for chunk_path in tqdm(chunk_path_list, total=chunk_length):
        if postfix == ".safetensors":
            chunk_state_dict = load_file(chunk_path, device="cpu")
        else:
            chunk_state_dict = torch.load(chunk_path, map_location="cpu")
        if "module" in chunk_state_dict.keys():
            chunk_state_dict = chunk_state_dict["module"]
        state_dict.update(chunk_state_dict)

    return state_dict

def pad_images(img_tensor, target_size, device) -> np.ndarray:
    """
    Pads an input image (as a NumPy array) to match the target aspect ratio,
    then resizes it to the target dimensions, preserving the original content's
    aspect ratio.

    The input NumPy array is expected to have the shape (C, H, W), where C=3 (RGB).

    Args:
        img_array: The input NumPy array (shape: C, H, W).
        target_w: The desired target width.
        target_h: The desired target height.

    Returns:
        The resized and padded NumPy array with shape (C, target_h, target_w).
    """
    
    target_w, target_h = target_size
    # 1. Input Validation and Conversion Setup
    if img_tensor.ndim != 3 or img_tensor.shape[0] != 3:
        raise ValueError("Input PyTorch Tensor must have shape (3, H, W) for RGB images.")

    # Convert PyTorch Tensor (C, H, W) to NumPy array (C, H, W)
    # .cpu() ensures it's on the CPU before conversion.
    img_array_float = img_tensor.cpu().numpy()
    # img_array_float *= 0  # TODO: debug
    # print(f" -----> {img_array_float.min()} | {img_array_float.max()}")

    # Denormalize from (-1, 1) to (0, 255) and convert to uint8 for PIL compatibility.
    # Formula: (x * 127.5) + 127.5
    img_array_np_uint8 = ((img_array_float + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    # img_array_np_uint8 = ((img_array_float * 127.5) + 127.5).clip(0, 255).astype(np.uint8)

    # The shape is (C, H, W). Extract H and W.
    _, origin_h, origin_w = img_array_np_uint8.shape

    # Convert NumPy array (C, H, W) -> (H, W, C) for PIL
    img_pil = Image.fromarray(np.transpose(img_array_np_uint8, (1, 2, 0)))

    # Calculate ratios
    target_ratio = target_w / target_h
    origin_ratio = origin_w / origin_h

    # 2. Determine padding direction and calculate new padded dimensions
    if origin_ratio < target_ratio:
        # Image is 'too tall' (height is too large relative to width)
        # Pad horizontally (left/right) to match the target ratio
        padded_w = math.ceil(origin_h * target_ratio)
        padded_h = origin_h
        padding_x = padded_w - origin_w
        padding_y = 0
    elif origin_ratio > target_ratio:
        # Image is 'too wide' (width is too large relative to height)
        # Pad vertically (top/bottom) to match the target ratio
        padded_w = origin_w
        padded_h = math.ceil(origin_w / target_ratio)
        padding_x = 0
        padding_y = padded_h - origin_h
    else:
        # Aspect ratios already match
        padded_w, padded_h = origin_w, origin_h
        padding_x = 0
        padding_y = 0

    # 3. Create the padded canvas (using black, which is 0 in 0-255 range, 
    # corresponding to -1 in the target -1 to 1 range)
    padded_img = Image.new(img_pil.mode, (padded_w, padded_h), color=(255, 255, 255))

    # 4. Paste the original image onto the center of the canvas
    paste_x = padding_x // 2
    paste_y = padding_y // 2
    padded_img.paste(img_pil, (paste_x, paste_y))

    # 5. Resize the padded image to the final target size
    final_img_pil = padded_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 6. Convert the final PIL Image back to NumPy array (H, W, C) -> (C, H, W)
    # This array is uint8 (0-255)
    final_img_array_hwc = np.array(final_img_pil)
    final_img_array_chw_uint8 = np.transpose(final_img_array_hwc, (2, 0, 1))

    # 7. Convert to float and Renormalize from (0, 255) back to (-1, 1)
    # Formula: (x / 127.5) - 1
    final_img_array_chw_float = final_img_array_chw_uint8.astype(np.float32)
    final_img_array_renormalized = (final_img_array_chw_float / 127.5) - 1.0

    # 8. Convert final NumPy array (C, H, W) back to PyTorch Tensor
    final_img_tensor = torch.from_numpy(final_img_array_renormalized).to(device=device)

    return final_img_tensor

class WanVideoEditPipeline:

    def __init__(
        self,
        args, 
        config,
        checkpoint_dir,
        high_noise_model_dir, 
        low_noise_model_dir, 
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False, 
        apply_rope_in_selfattn=False, 
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.args = args
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False
        
        shard_fn = partial(shard_model, device_id=device_id)
        
        if args.text_encoder == "t5":
            self.text_encoder = T5EncoderModel(
                # text_len=config.text_len,
                text_len=T5_TEXT_LEN,
                dtype=config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
                shard_fn=shard_fn if t5_fsdp else None
            )
            self.tokenizer = self.text_encoder.tokenizer
        else:
            raise ValueError(f"Unrecognized text encoder {args.text_encoder}")

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)
        
        # Build DiT models
        logging.info(f"🔄 Building DiT models")
        self.high_noise_model, self.low_noise_model = build_dit_models(args)

        # Build low_noise_model
        logging.info(f"🔄 loading low_noise_model from: {low_noise_model_dir}")
        # low_noise_state_dict = load_state_dict(low_noise_model_dir, postfix=".safetensors" if not args.metaquery_enabled else ".bin")
        low_noise_state_dict = load_state_dict(low_noise_model_dir, postfix=".bin")
        logging.info(f"✅ low_noise_model loaded")
        m, u = self.low_noise_model.load_state_dict(low_noise_state_dict, strict=False)
        for _m in m:
            logging.info(f"⚠️ Missing {_m} low_noise_model")
        for _u in u:
            logging.info(f"⚠️ Unexpect {_u} low_noise_model")

        # meta-query方案为了节省推理显存，把模型中和时间步无关的模块放到外面来
        if args.text_encoder == "t5" and args.metaquery_enabled:
            try:
                self.low_noise_vlm_metaquery = getattr(self.low_noise_model, "vlm_metaquery")
                self.low_noise_vlm_encoder = getattr(self.low_noise_model, "vlm_encoder")
                self.low_noise_vlm_connector = getattr(self.low_noise_model, "vlm_connector")
                setattr(self.low_noise_model, "vlm_metaquery", None)
                setattr(self.low_noise_model, "vlm_encoder", None)
                setattr(self.low_noise_model, "vlm_connector", None)
                self.low_noise_vlm_metaquery.to(torch.device("cpu"))
                self.low_noise_vlm_encoder.to(torch.device("cpu"))
                self.low_noise_vlm_connector.to(torch.device("cpu"))
                torch.cuda.empty_cache()
            except:
                self.low_noise_vlm_metaquery = None
                self.low_noise_vlm_encoder = None
                self.low_noise_vlm_connector = None
        
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype, 
            configure_type="edit" if args.dit_version_low != "raw" else "raw", 
            dit_version=args.dit_version_low)

        # Build high_noise_model
        logging.info(f"🔄 loading high_noise_model from: {high_noise_model_dir}")
        # high_noise_state_dict = load_state_dict(high_noise_model_dir, postfix=".safetensors" if not args.metaquery_enabled else ".bin")
        high_noise_state_dict = load_state_dict(high_noise_model_dir, postfix=".bin")
        logging.info(f"✅ high_noise_model loaded")
        m, u = self.high_noise_model.load_state_dict(high_noise_state_dict, strict=False)
        for _m in m:
            logging.info(f"⚠️ Missing {_m} high_noise_model")
        for _u in u:
            logging.info(f"⚠️ Unexpect {_u} high_noise_model")

        # meta-query方案为了节省推理显存，把模型中和时间步无关的模块放到外面来
        if args.text_encoder == "t5" and args.metaquery_enabled:
            self.high_noise_vlm_metaquery = getattr(self.high_noise_model, "vlm_metaquery")
            self.high_noise_vlm_encoder = getattr(self.high_noise_model, "vlm_encoder")
            self.high_noise_vlm_connector = getattr(self.high_noise_model, "vlm_connector")
            setattr(self.high_noise_model, "vlm_metaquery", None)
            setattr(self.high_noise_model, "vlm_encoder", None)
            setattr(self.high_noise_model, "vlm_connector", None)
            self.high_noise_vlm_metaquery.to(torch.device("cpu"))
            self.high_noise_vlm_encoder.to(torch.device("cpu"))
            self.high_noise_vlm_connector.to(torch.device("cpu"))
            torch.cuda.empty_cache()
        
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype, 
            configure_type="edit" if args.dit_version_high != "raw" else "raw", 
            dit_version=args.dit_version_high)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype, configure_type, dit_version):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                if configure_type == "raw":
                    block.self_attn.forward = types.MethodType(
                        sp_attn_forward, block.self_attn)
                elif configure_type == "edit":
                    block.self_attn.forward = types.MethodType(
                        sp_attn_forward_edit, block.self_attn)
                else:
                    raise ValueError

                if hasattr(block, "hierarchical_subjects_fusion"):
                    block.hierarchical_subjects_fusion.forward = types.MethodType(
                        sp_hier_fusion_forward, block.hierarchical_subjects_fusion)
                    # block.hierarchical_subjects_fusion.intra_attn.forward = types.MethodType()
                    # block.hierarchical_subjects_fusion.inter_attn.forward = types.MethodType()
            
            if configure_type == "raw":
                model.forward = types.MethodType(sp_dit_forward, model)
            elif configure_type == "edit":
                if dit_version == "v1":
                    model.forward = types.MethodType(sp_dit_forward_edit_v1, model)
                elif dit_version == "v1_h":
                    model.forward = types.MethodType(sp_dit_forward_edit_v1_high, model)
                elif dit_version == "v1_l":
                    model.forward = types.MethodType(sp_dit_forward_edit_v1_low, model)
                elif dit_version == "v3_ct":
                    model.forward = types.MethodType(sp_dit_forward_edit_v3_ct, model)
                else:
                    raise ValueError(f"Unrecognized dit version {dit_version}")
            else:
                raise ValueError

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name), required_model_name

    def encode_t5(self, input_prompt):
        context = self.text_encoder([input_prompt], self.device)
        context = [t.to(self.device) for t in context]
        return context

    @torch.no_grad()
    def encode_metaquery(self, input_prompt, condition_images, ref_ids, moe_stage):
        context = {"video_prompt": input_prompt}
        if ref_ids is not None:
            context["ref_id"] = ref_ids
        if "ref_img" in condition_images.keys():
            if isinstance(condition_images["ref_img"], list):
                context["ref_pixel_values"] = torch.stack([img.unsqueeze(0) for img in condition_images["ref_img"]], dim=2)  # (b*n c 1 h w)
            else:
                context["ref_pixel_values"] = condition_images["ref_img"].unsqueeze(0).unsqueeze(2)
        if "ff_img" in condition_images.keys():
            context["ff_pixel_value"] = condition_images["ff_img"].unsqueeze(0).unsqueeze(2)
        if "lf_img" in condition_images.keys():
            context["lf_pixel_value"] = condition_images["ref_img"].unsqueeze(0).unsqueeze(2)
            
        # High-noise model
        if moe_stage == "high_noise_model":
            self.high_noise_vlm_metaquery.to(self.device)
            self.high_noise_vlm_encoder.to(self.device)
            self.high_noise_vlm_connector.to(self.device)
            # Meta-query
            vlm_context_emb, vlm_attention_mask = self.high_noise_vlm_metaquery(
                video_prompts=context["video_prompt"], 
                context=context, 
                padding_info=None, 
            )
            # query-encoder & connector
            vlm_context_emb = self.high_noise_vlm_encoder(vlm_context_emb)
            vlm_context_emb = self.high_noise_vlm_connector(vlm_context_emb)
            self.high_noise_vlm_metaquery.to(torch.device("cpu"))
            self.high_noise_vlm_encoder.to(torch.device("cpu"))
            self.high_noise_vlm_connector.to(torch.device("cpu"))
            torch.cuda.empty_cache()
        # Low-noise model
        elif moe_stage == "low_noise_model":
            try:
                self.low_noise_vlm_metaquery.to(self.device)
                self.low_noise_vlm_encoder.to(self.device)
                self.low_noise_vlm_connector.to(self.device)
                # Meta-query
                vlm_context_emb, vlm_attention_mask = self.low_noise_vlm_metaquery(
                    video_prompts=context["video_prompt"], 
                    context=context, 
                    padding_info=None, 
                )
                # query-encoder & connector
                vlm_context_emb = self.low_noise_vlm_encoder(vlm_context_emb)
                vlm_context_emb = self.low_noise_vlm_connector(vlm_context_emb)
                self.low_noise_vlm_metaquery.to(torch.device("cpu"))
                self.low_noise_vlm_encoder.to(torch.device("cpu"))
                self.low_noise_vlm_connector.to(torch.device("cpu"))
                torch.cuda.empty_cache()
            except:
                vlm_context_emb = None

        return vlm_context_emb

    def generate_complex(self,
                 input_prompt,
                 input_comprehensive_prompt,
                 input_imgs,
                 ref_ids=None,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale_text=5.0,
                 guide_scale_img=0.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 specify_size=False,
                 one_step_denoising=False,
                 **kwargs):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        guide_scale_text = (guide_scale_text, guide_scale_text) if isinstance(
            guide_scale_text, float) else guide_scale_text
        guide_scale_img = (guide_scale_img, guide_scale_img) if isinstance(
            guide_scale_img, float) else guide_scale_img

        # preprocess condition imgs
        condition_imgs, nul_condition_imgs = {}, {}
        for k, img in input_imgs.items():
            if isinstance(img, list):
                condition_imgs[k] = [TF.to_tensor(img[idx]).sub_(0.5).div_(0.5).to(self.device) 
                                     for idx in range(len(img))]
                nul_condition_imgs[k] = [torch.zeros_like(TF.to_tensor(img[idx])).sub_(1.0).to(self.device) 
                                         for idx in range(len(img))]
            else:
                condition_imgs[k] = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
                nul_condition_imgs[k] = torch.zeros_like(TF.to_tensor(img)).sub_(1.0).to(self.device)

        # condition_imgs = {k: TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device) for k, img in input_imgs.items()}
        # nul_condition_imgs = {k: torch.zeros_like(TF.to_tensor(img)).sub_(0.5).div_(0.5).to(self.device) for k, img in input_imgs.items()}
        F = frame_num
        h, w = size
        if not specify_size:
            # size of generated video is the same as condition images
            for _, img in condition_imgs.items():
                if isinstance(img, list):
                    h, w = img[0].shape[1:]
                else:
                    h, w = img.shape[1:]
        else:
            # pad condition images
            for key, img in condition_imgs.items():
                if isinstance(img, list):
                    for idx in range(len(img)):
                        condition_imgs[key][idx] = pad_images(img[idx], (w, h), device=self.device)
                else:
                    condition_imgs[key] = pad_images(img, (w, h), device=self.device)
            for key, img in nul_condition_imgs.items():
                if isinstance(img, list):
                    for idx in range(len(img)):
                        nul_condition_imgs[key][idx] = pad_images(img[idx], (w, h), device=self.device)
                else:
                    nul_condition_imgs[key] = pad_images(img, (w, h), device=self.device)
        
        # # Resize the input images
        # def resize_input_images(input_images):
        #     import math
        #     scale = 1.0 / math.sqrt(2)
        #     resized = {key: [] for key in input_images.keys()}
        #     for key, imgs in input_images.items():
        #         for img in imgs:
        #             h,w = img.shape[-2:]
        #             new_w = int(w * scale)
        #             new_h = int(h * scale)
        #             # resized[key].append(img.resize((new_w, new_h), Image.LANCZOS))
        #             resize_img = torch.nn.functional.interpolate(img.unsqueeze(0), (new_h,new_w), mode='bilinear').squeeze(0)
        #             resized[key].append(resize_img)
        #     return resized

        # condition_imgs = resize_input_images(condition_imgs)
        # nul_condition_imgs = resize_input_images(nul_condition_imgs)

        if not specify_size:
            aspect_ratio = h / w
            max_area = size[0] * size[1]
            lat_h = round(
                np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
                (self.patch_size[1] * 4) * (self.patch_size[1] * 4))        # Make sure the seq_len is divisible by sp_size = 8
            lat_w = round(
                np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
                (self.patch_size[2] * 4) * (self.patch_size[2] * 4))        # Make sure the seq_len is divisible by sp_size = 8
            h = lat_h * self.vae_stride[1]
            w = lat_w * self.vae_stride[2]
        else:
            lat_h = h // 8
            lat_w = w // 8
        size = [w, h]

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        logging.info(f"Target shape: {target_shape}")

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if kwargs.get("correct_id_reference", False):
            num_ref = len(condition_imgs["ref_img"])
            for idx in range(num_ref):
                input_prompt += f" image of [PERSON_{idx+1}] is: "
                n_prompt += f" image of [PERSON_{idx+1}] is: "
            print(f"✅ 【Positive Prompt】{input_prompt}")
            print(f"✅ 【Negative Prompt】{n_prompt}")

        if not self.t5_cpu:
            if self.args.text_encoder == "t5":
                self.text_encoder.model.to(self.device)
                # encode vlm embeding
                context = self.encode_t5(input_comprehensive_prompt)
                _, attention_mask = self.tokenizer([input_comprehensive_prompt], return_mask=True, add_special_tokens=True)
                # print(f" >>>>> DEBUG, t5_prompt: ''")
                # context = self.encode_t5("")        # DEBUG
                # _, attention_mask = self.tokenizer([n_prompt], return_mask=True, add_special_tokens=True) # DEBUG
                context_null = self.encode_t5(n_prompt)
                _, nul_attention_mask = self.tokenizer([n_prompt], return_mask=True, add_special_tokens=True)
                if offload_model:
                    self.text_encoder.model.cpu()
                if self.args.metaquery_enabled:
                    high_noise_vlm_context_emb = self.encode_metaquery(
                        [input_prompt], condition_imgs, ref_ids, "high_noise_model")
                    nul_high_noise_vlm_context_emb = self.encode_metaquery(
                        [n_prompt], nul_condition_imgs, ref_ids, "high_noise_model")
                    low_noise_vlm_context_emb = self.encode_metaquery(
                        [input_prompt], condition_imgs, ref_ids, "low_noise_model")
                    nul_low_noise_vlm_context_emb = self.encode_metaquery(
                        [n_prompt], nul_condition_imgs, ref_ids, "low_noise_model")
            elif self.args.text_encoder == "qwenvl-2.5":
                pass
            elif self.args.text_encoder == "mmt5":
                pass
            elif self.args.text_encoder == "mmt5-mm":
                pass
            elif self.args.text_encoder == "mmt5-omni":
                pass
            else:
                raise ValueError(f"Unrecognized text_encoder {args.text_encoder}")
        else:
            if self.args.text_encoder == "t5":
                # encode vlm embeding
                context = self.encode_t5(input_comprehensive_prompt)
                _, attention_mask = self.tokenizer([input_comprehensive_prompt], return_mask=True, add_special_tokens=True)
                # print(f" >>>>> DEBUG, t5_prompt: ''")
                # context = self.encode_t5("")    # DEBUG
                context_null = self.encode_t5(n_prompt)
                _, nul_attention_mask = self.tokenizer([n_prompt], return_mask=True, add_special_tokens=True)
                if self.args.metaquery_enabled:
                    high_noise_vlm_context_emb = self.encode_metaquery(
                        [input_prompt], condition_imgs, ref_ids, "high_noise_model")
                    nul_high_noise_vlm_context_emb = self.encode_metaquery(
                        [n_prompt], nul_condition_imgs, ref_ids, "high_noise_model")
                    low_noise_vlm_context_emb = self.encode_metaquery(
                        [input_prompt], condition_imgs, ref_ids, "low_noise_model")
                    nul_low_noise_vlm_context_emb = self.encode_metaquery(
                        [n_prompt], nul_condition_imgs, ref_ids, "low_noise_model")
            elif self.args.text_encoder == "qwenvl-2.5":
                pass
            elif self.args.text_encoder == "mm5":
                pass
            elif self.args.text_encoder == "mmt5-mm":
                pass
            elif self.args.text_encoder == "mmt5-omni":
                pass
            else:
                raise ValueError(f"Unrecognized text_encoder {args.text_encoder}")

        print(f"📊📊📊 Number of Valid T5 Tokens: {attention_mask.sum().item()}")

        condition_latents, nul_condition_latents = {}, {}
        for k, img in condition_imgs.items():
            if isinstance(img, list):
                h_, w_ = img[0].shape[-2:]
                condition_latents[k] = torch.cat([self.vae.encode(
                    torch.nn.functional.interpolate(
                        img[idx][None], size=(h_, w_), mode="bicubic"
                    ).transpose(0, 1).unsqueeze(0))[0].unsqueeze(0) 
                    for idx in range(len(img))], dim=0)
            else:
                h_, w_ = img.shape[-2:]
                condition_latents[k] = self.vae.encode(
                    torch.nn.functional.interpolate(
                        img[None], size=(h_, w_), mode="bicubic"
                    ).transpose(0, 1).unsqueeze(0)
                )[0].unsqueeze(0)
        for k, img in nul_condition_imgs.items():
            if isinstance(img, list):
                h_, w_ = img[0].shape[-2:]
                nul_condition_latents[k] = torch.cat([self.vae.encode(
                    torch.nn.functional.interpolate(
                        img[idx][None], size=(h_, w_), mode="bicubic"
                    ).transpose(0, 1).unsqueeze(0))[0].unsqueeze(0) 
                    for idx in range(len(img))], dim=0)
            else:
                h_, w_ = img.shape[-2:]
                nul_condition_latents[k] = self.vae.encode(
                    torch.nn.functional.interpolate(
                        img[None], size=(h_, w_), mode="bicubic"
                    ).transpose(0, 1).unsqueeze(0)
                )[0].unsqueeze(0)

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps
            logging.info(f"Shift = {shift}")
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            if context is not None:
                context_it = {"text_context": context, "ref_id": ref_ids}
                context_i = {"text_context": context_null, "ref_id": ref_ids}
            else:
                context_it = {"ref_id": ref_ids}
                context_i = {"ref_id": ref_ids}
            
            if self.args.text_encoder == "t5":
                context_it["text_attention_mask"] = attention_mask
                context_i["text_attention_mask"] = nul_attention_mask
            elif self.args.text_encoder == "mmt5":
                pass
            elif self.args.text_encoder == "mmt5-mm":
                pass
            elif self.args.text_encoder == "mmt5-omni":
                pass
            
            # TODO: 【注释掉】测试text_encoder+T2V的效果
            if not self.args.textencoder_only:
                context_it.update({k.replace("_img", "_context"): v for k, v in condition_latents.items()})
                context_i.update({k.replace("_img", "_context"): v for k, v in condition_latents.items()})
            
            if context_null is not None:
                context_neg = {"text_context": context_null, "ref_id": ref_ids}
            else:
                context_neg = {"ref_id": ref_ids}
                
            if self.args.text_encoder == "t5":
                context_neg["text_attention_mask"] = nul_attention_mask
            elif self.args.text_encoder == "mmt5":
                pass
            elif self.args.text_encoder == "mmt5-mm":
                pass
            elif self.args.text_encoder == "mmt5-omni":
                pass
            
            # TODO: 【注释掉】测试text_encoder+T2V的效果
            if not self.args.textencoder_only:
                context_neg.update({k.replace("_img", "_context"): v for k, v in nul_condition_latents.items()})
            
            if self.args.metaquery_enabled:
                context_it["video_prompt"] = input_prompt
                if "ref_img" in condition_imgs.keys():
                    if isinstance(condition_imgs["ref_img"], list):
                        context_it["ref_pixel_values"] = torch.stack([img.unsqueeze(1) for img in condition_imgs["ref_img"]], dim=2)  # (b*n c 1 h w)
                        context_i["ref_pixel_values"] = torch.stack([img.unsqueeze(1) for img in condition_imgs["ref_img"]], dim=2)  # (b*n c 1 h w)
                    else:
                        context_it["ref_pixel_values"] = condition_imgs["ref_img"].unsqueeze(0).unsqueeze(2)
                        context_i["ref_pixel_values"] = condition_imgs["ref_img"].unsqueeze(0).unsqueeze(2)
                if "ff_img" in condition_imgs.keys():
                    context_it["ff_pixel_value"] = condition_imgs["ff_img"].unsqueeze(0).unsqueeze(2)
                    context_i["ff_pixel_value"] = condition_imgs["ff_img"].unsqueeze(0).unsqueeze(2)
                if "lf_img" in condition_imgs.keys():
                    context_it["lf_pixel_value"] = condition_imgs["ref_img"].unsqueeze(0).unsqueeze(2)
                    context_i["lf_pixel_value"] = condition_imgs["ref_img"].unsqueeze(0).unsqueeze(2)
                
                context_neg["video_prompt"] = n_prompt
                if "ref_img" in nul_condition_imgs.keys():
                    if isinstance(nul_condition_imgs["ref_img"], list):
                        context_neg["ref_pixel_values"] = torch.stack([img.unsqueeze(1) for img in nul_condition_imgs["ref_img"]], dim=2)  # (b*n c 1 h w)
                    else:
                        context_neg["ref_pixel_values"] = nul_condition_imgs["ref_img"].unsqueeze(0).unsqueeze(2)
                if "ff_img" in nul_condition_imgs.keys():
                    context_neg["ff_pixel_value"] = nul_condition_imgs["ff_img"].unsqueeze(0).unsqueeze(2)
                if "lf_img" in nul_condition_imgs.keys():
                    context_neg["lf_pixel_value"] = nul_condition_imgs["ref_img"].unsqueeze(0).unsqueeze(2)

            if kwargs.get("face_masks", None) is not None:
                from einops import rearrange
                face_masks = rearrange(kwargs.get("face_masks"), "h w n -> 1 n 1 h w")  # (b c f h w)
                face_masks = face_masks.repeat(1, 1, F, 1, 1).to(device=self.device, dtype=torch.float32)
                context_it["ip_mask"] = face_masks
                context_i["ip_mask"] = face_masks
                context_neg["ip_mask"] = face_masks

            context_i["crop_bbox"] = kwargs.get("crop_bbox", None)
            context_it["crop_bbox"] = kwargs.get("crop_bbox", None)
            context_neg["crop_bbox"] = kwargs.get("crop_bbox", None)

            arg_c = {
                "context": context_it, 
                'seq_len': seq_len, 
                "use_gradient_checkpointing": True, 
                "slg": -1, 
            }
            arg_null = {
                "context": context_neg, 
                'seq_len': seq_len, 
                "use_gradient_checkpointing": True, 
                "slg": kwargs.get("slg", -1),
            }

            pbar = tqdm(timesteps, desc="Video Generation")
            for _, t in enumerate(timesteps):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                model, required_model_name = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                
                sample_guide_scale_text = guide_scale_text[1] if t.item(
                ) >= boundary else guide_scale_text[0]

                sample_guide_scale_img = guide_scale_img[1] if t.item(
                ) >= boundary else guide_scale_img[0]

                # MMT5-MM方案节省推理时显存的策略
                if self.args.text_encoder == "mmt5-mm":
                    pass
                elif self.args.text_encoder == "t5" and self.args.metaquery_enabled:
                    if t.item() > boundary:
                        arg_it = {
                            "context": context_it, 
                            'seq_len': seq_len, 
                            "use_gradient_checkpointing": True, 
                            "slg": -1, 
                        }
                        arg_it["context"]["vlm_context_emb"] = high_noise_vlm_context_emb
                        arg_i = {
                            "context": context_i, 
                            'seq_len': seq_len, 
                            "use_gradient_checkpointing": True, 
                            "slg": -1, 
                        }
                        arg_i["context"]["vlm_context_emb"] = high_noise_vlm_context_emb
                        arg_neg = {
                            "context": context_neg, 
                            'seq_len': seq_len, 
                            "use_gradient_checkpointing": True, 
                            "slg": -1, 
                        }
                        arg_neg["context"]["vlm_context_emb"] = nul_high_noise_vlm_context_emb
                    elif t.item() > boundary / 2:
                        arg_it = {
                            "context": context_it, 
                            'seq_len': seq_len, 
                            "use_gradient_checkpointing": True, 
                            "slg": -1, 
                        }
                        arg_it["context"]["vlm_context_emb"] = low_noise_vlm_context_emb
                        arg_i = {
                            "context": context_i, 
                            'seq_len': seq_len, 
                            "use_gradient_checkpointing": True, 
                            "slg": -1, 
                        }
                        arg_i["context"]["vlm_context_emb"] = low_noise_vlm_context_emb
                        arg_neg = {
                            "context": context_neg, 
                            'seq_len': seq_len, 
                            "use_gradient_checkpointing": True, 
                            "slg": -1, 
                        }
                        arg_neg["context"]["vlm_context_emb"] = nul_low_noise_vlm_context_emb
                    else:
                        arg_it = {
                            "context": context_it, 
                            'seq_len': seq_len, 
                            "use_gradient_checkpointing": True, 
                            "slg": -1, 
                        }
                        arg_it["context"]["vlm_context_emb"] = low_noise_vlm_context_emb
                        arg_i = {
                            "context": context_i, 
                            'seq_len': seq_len, 
                            "use_gradient_checkpointing": True, 
                            "slg": -1, 
                        }
                        arg_i["context"]["vlm_context_emb"] = low_noise_vlm_context_emb
                        arg_neg = {
                            "context": context_i, 
                            'seq_len': seq_len, 
                            "use_gradient_checkpointing": True, 
                            "slg": -1, 
                        }
                        arg_neg["context"]["vlm_context_emb"] = nul_low_noise_vlm_context_emb

                noise_pred_it = model(
                    latent_model_input, t=timestep, **arg_it)[0]
                noise_pred_i = model(
                    latent_model_input, t=timestep, **arg_i)[0]
                
                if sample_guide_scale_img > 0:
                    noise_pred_neg = model(
                        latent_model_input, t=timestep, **arg_neg)[0]
                    noise_pred = noise_pred_neg + sample_guide_scale_img * (noise_pred_i - noise_pred_neg) + \
                        sample_guide_scale_text * (noise_pred_it - noise_pred_i) # neg + 7.5 it - 5 neg - 2.5 i
                else:
                    noise_pred = noise_pred_i + sample_guide_scale_text * (noise_pred_it - noise_pred_i)

                # High-noise denoising: after finishing high-noise stage, directly predict x0 and exit
                if one_step_denoising:
                    if t.item() > boundary:
                        # Still in high-noise stage, do normal denoising
                        temp_x0 = sample_scheduler.step(
                            noise_pred.unsqueeze(0),
                            t,
                            latents[0].unsqueeze(0),
                            return_dict=False,
                            generator=seed_g)[0]
                        latents = [temp_x0.squeeze(0)]
                        pbar.update(1)
                        pbar.set_postfix(timestep=f"{t.item():02f}", moe_stage=required_model_name, info="high_noise_step")
                        continue
                    else:
                        # High-noise stage done (t <= boundary), directly predict x0
                        # For flow matching: x_t = (1 - sigma) * x_0 + sigma * eps
                        # Model predicts velocity v = eps - x_0
                        # Therefore: x_0 = x_t - sigma * v
                        sigma = t.item() / 1000.0
                        x_t = latents[0]
                        x0_pred = x_t - sigma * noise_pred
                        logging.info(
                            f"[High-noise denoising DEBUG] boundary={boundary}, timestep={t.item():.2f}, sigma={sigma:.4f}, "
                            f"x_t: mean={x_t.mean().item():.4f}, std={x_t.std().item():.4f}, "
                            f"noise_pred: mean={noise_pred.mean().item():.4f}, std={noise_pred.std().item():.4f}, "
                            f"x0_pred: mean={x0_pred.mean().item():.4f}, std={x0_pred.std().item():.4f}"
                        )
                        latents = [x0_pred]
                        pbar.update(1)
                        pbar.set_postfix(timestep=f"{t.item():02f}", moe_stage=required_model_name, info="high_noise_x0")
                        break

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

                pbar.update(1)
                pbar.set_postfix(timestep=f"{t.item():02f}", moe_stage=required_model_name)

            pbar.close()

            x0 = latents
            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None


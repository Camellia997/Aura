# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import json
import logging
import os
import sys
sys.path.append(os.getcwd())
import random
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from src.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from src.utils.utils import save_video, str2bool
from src.pipelines.hyvideo_edit_pipeline import WanVideoEditPipeline

# Fixed configuration for the opensource inference package.
TASK = "t2v-A14B"
TEXT_ENCODER = "t5"
DIT_VERSION_HIGH = "v1_h"
DIT_VERSION_LOW = "v1_l"


def _validate_args(args):
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert TASK in WAN_CONFIGS, f"Unsupported task: {TASK}"

    cfg = WAN_CONFIGS[TASK]
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    assert args.size in SUPPORTED_SIZES[TASK], \
        f"Unsupported size {args.size} for task {TASK}, supported sizes are: {', '.join(SUPPORTED_SIZES[TASK])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="HunyuanVideo edit inference (T5 + metaquery, high/low-noise DiT)")

    # Generation settings
    parser.add_argument("--size", type=str, default="720*1280",
                        choices=list(SIZE_CONFIGS.keys()),
                        help="The area (width*height) of the generated video.")
    parser.add_argument("--frame_num", type=int, default=None,
                        help="How many frames to generate. Should be 4n+1.")
    parser.add_argument("--sample_steps", type=int, default=None, help="Sampling steps.")
    parser.add_argument("--sample_solver", type=str, default='unipc',
                        choices=['unipc', 'dpm++'], help="Solver used to sample.")
    parser.add_argument("--sample_shift", type=float, default=None,
                        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument("--boundary", type=float, default=0.875,
                        help="Boundary for switching between high_noise and low_noise models.")
    parser.add_argument("--base_seed", type=int, default=1024, help="Random seed.")
    parser.add_argument("--gen_mode", type=str, default="ref",
                        choices=["ff", "lf", "ref", "nul"], help="Generation mode.")
    parser.add_argument("--slg", type=int, default=-1, help="SLG layer (-1 disables).")
    parser.add_argument("--guide_scale_text", type=float, nargs=2, default=[5.0, 5.0],
                        metavar=("LOW", "HIGH"),
                        help="Text CFG scale for (low_noise, high_noise) models.")
    parser.add_argument("--guide_scale_img", type=float, nargs=2, default=[5.0, 3.0],
                        metavar=("LOW", "HIGH"),
                        help="Image CFG scale for (low_noise, high_noise) models.")

    # Model / weight paths
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Base checkpoint directory (T5 + VAE).")
    parser.add_argument("--high_noise_model_dir", type=str, default="",
                        help="Directory of high noise DiT weights.")
    parser.add_argument("--low_noise_model_dir", type=str, default="",
                        help="Directory of low noise DiT weights.")
    parser.add_argument("--vlm_dir", type=str, default=None,
                        help="Path/HF id of the Qwen2.5-VL model used by the metaquery encoder.")

    # Data / IO
    parser.add_argument("--csv_file", type=str, default="", help="Validation json file.")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save videos.")
    parser.add_argument("--log_file", type=str, default="results/logging.txt", help="Output log file.")

    # Distributed / performance
    parser.add_argument("--ulysses_size", type=int, default=1,
                        help="Size of the ulysses (sequence) parallelism in DiT.")
    parser.add_argument("--t5_fsdp", action="store_true", default=False,
                        help="Use FSDP for T5.")
    parser.add_argument("--t5_cpu", action="store_true", default=False,
                        help="Place T5 model on CPU.")
    parser.add_argument("--dit_fsdp", action="store_true", default=False,
                        help="Use FSDP for DiT.")
    parser.add_argument("--offload_model", type=str2bool, default=None,
                        help="Offload models to CPU between forwards to save VRAM.")
    parser.add_argument("--convert_model_dtype", action="store_true", default=False,
                        help="Convert model parameters dtype to config.param_dtype.")
    parser.add_argument("--apply_rope_in_selfattn", type=str2bool, default="False",
                        help="Whether to apply rope inside selfattn.")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes.")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of the current node.")

    args = parser.parse_args()

    # Inject fixed configuration expected by the pipeline.
    args.task = TASK
    args.text_encoder = TEXT_ENCODER
    args.dit_version_high = DIT_VERSION_HIGH
    args.dit_version_low = DIT_VERSION_LOW
    args.metaquery_enabled = True
    args.textencoder_only = False
    args.return_multi_layer_states = False
    args.return_face_mask = False
    args.cfg_strategy = "complex"

    _validate_args(args)
    return args


def _init_logging(rank, log_file="application.log"):
    if rank == 0:
        log_dir = os.path.split(log_file)[0]
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            filename=log_file,
            filemode="a")
    else:
        logging.basicConfig(level=logging.ERROR)


def init_distributed_group():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')


def merge_images_horizontally(*images):
    """Horizontally concatenate images, matching all heights to the first image."""
    if not images:
        return None
    if len(images) == 1:
        return images[0].copy()

    target_height = images[0].height
    processed_images = []
    for img in images:
        temp_img = img.copy()
        if temp_img.height != target_height:
            scale_ratio = target_height / temp_img.height
            new_width = int(temp_img.width * scale_ratio)
            temp_img = temp_img.resize((new_width, target_height), Image.Resampling.LANCZOS)
        processed_images.append(temp_img)

    total_width = sum(img.width for img in processed_images)
    result_image = Image.new(processed_images[0].mode, (total_width, target_height))
    x_offset = 0
    for img in processed_images:
        result_image.paste(img, (x_offset, 0))
        x_offset += img.width
    return result_image


def resize_and_crop_scene_image(image, target_width, target_height):
    """Resize and center-crop to the target resolution while preserving aspect ratio."""
    import cv2
    image = np.asarray(image)
    orig_h, orig_w = image.shape[:2]

    scale = max(target_width / orig_w, target_height / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    start_x = (new_w - target_width) // 2
    start_y = (new_h - target_height) // 2
    cropped_image = resized_image[start_y:start_y + target_height,
                                  start_x:start_x + target_width]
    return Image.fromarray(cropped_image)


def generate_face_masks(num_ref, target_shape, height_ratio=0.3, width_ratio=0.15):
    target_width, target_height = target_shape
    half_width = target_width // 2
    face_masks = np.zeros((target_height, target_width, 2)).astype(np.float32)
    for n in range(num_ref):
        if n == 0:
            ctr_x = half_width // 2
            ctr_y = target_height // 4
        elif n == 1:
            ctr_x = half_width + half_width // 2
            ctr_y = target_height // 4
        else:
            raise ValueError("Only supports two IPs")
        x_min = max(0, ctr_x - int(target_width * width_ratio))
        x_max = min(target_width, ctr_x + int(target_width * width_ratio))
        y_min = max(0, ctr_y - int(target_height * height_ratio))
        y_max = min(target_height, ctr_y + int(target_height * height_ratio))
        face_masks[y_min:y_max, x_min:x_max, n] = 1
    return face_masks


def pad_image(image, skip=False):
    """Pad white borders on the left/right sides (each 1/3 of the width)."""
    if skip:
        return image
    w, h = image.size
    pad_w = w // 3
    new_w = w + pad_w * 2
    padded = Image.new("RGB", (new_w, h), (255, 255, 255))
    padded.paste(image, (pad_w, 0))
    return padded


def batch_generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank, log_file=args.log_file)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), \
            "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (args.ulysses_size > 1), \
            "sequence parallel is not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, \
            "The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]
    cfg.boundary = args.boundary
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, \
            f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info("Creating WanVideoEditPipeline pipeline.")
    wan_t2v = WanVideoEditPipeline(
        args=args,
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        high_noise_model_dir=args.high_noise_model_dir,
        low_noise_model_dir=args.low_noise_model_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        apply_rope_in_selfattn=args.apply_rope_in_selfattn,
    )

    validation_df = json.load(open(args.csv_file, "r"))
    logging.info(f"Validation set has {len(validation_df)} files")

    # ----- Multi-node data sharding -----
    num_nodes = args.nodes
    node_rank = args.node_rank
    total_samples = len(validation_df)
    node_indices = list(range(node_rank, total_samples, num_nodes))
    validation_df = [validation_df[i] for i in node_indices]
    logging.info(f"Node {node_rank}/{num_nodes}: {len(validation_df)} samples after inter-node sharding")

    # ----- Intra-node data-parallel sharding -----
    sp_size = args.ulysses_size if args.ulysses_size > 1 else world_size
    dp_size = max(world_size // sp_size, 1)
    dp_rank = rank // sp_size
    if dp_size > 1:
        dp_indices = list(range(dp_rank, len(validation_df), dp_size))
        validation_df = [validation_df[i] for i in dp_indices]
    logging.info(f"Node {node_rank}, DP rank {dp_rank}/{dp_size}: {len(validation_df)} samples after intra-node sharding")

    for idx in range(len(validation_df)):
        videoid = str(validation_df[idx]["videoid"])
        base_seed = int(validation_df[idx].get("seed", args.base_seed))
        ff_image_path = validation_df[idx]["ff_image_path"]
        lf_image_path = ff_image_path
        ref_image_path = validation_df[idx]["ref_image_path"]
        mul_ref_image_paths = validation_df[idx]["mul_ref_image_path"]
        raw_crop_bbox = validation_df[idx].get("crop_bbox", [[-1, -1, -1, -1]])
        ref_ids = list()

        prompt = validation_df[idx].get("prompt", "")
        if "enhanced_prompt_id_reference" in validation_df[idx].keys():
            prompt = validation_df[idx]["enhanced_prompt_id_reference"]["visual_description"][0]
            if len(validation_df[idx]["enhanced_prompt_id_reference"]["visual_description"]) == 2:
                comprehensive_prompt = validation_df[idx]["enhanced_prompt_id_reference"]["visual_description"][1]
            else:
                comprehensive_prompt = prompt
            correct_id_reference = False
        else:
            correct_id_reference = False
            comprehensive_prompt = prompt

        video_path = validation_df[idx].get("video_path", None)
        mask_path = validation_df[idx].get("mask_path", None)
        videoid = videoid.split(".")[0]

        if os.path.exists(f"{args.save_dir}/{videoid}.mp4"):
            continue

        # Read img
        if ff_image_path is not None and args.gen_mode == "ff":
            ff_img = Image.open(ff_image_path).convert("RGB")
        elif ff_image_path is None and args.gen_mode == "ff":
            continue

        if lf_image_path is not None and args.gen_mode == "lf":
            lf_img = Image.open(lf_image_path).convert("RGB")
        elif lf_image_path is None and args.gen_mode == "lf":
            continue

        if (ref_image_path is not None or mul_ref_image_paths is not None) and args.gen_mode == "ref":
            if ref_image_path is not None:
                ref_img = Image.open(ref_image_path).convert("RGB")
            if mul_ref_image_paths is not None:
                mul_ref_imgs = [Image.open(path).convert("RGB") for path in mul_ref_image_paths]

            if "ref_ids" in validation_df[idx].keys():
                ref_ids = validation_df[idx]["ref_ids"]["person"][:len(mul_ref_imgs)]
            else:
                if mul_ref_image_paths is None:
                    ref_ids.append(1)
                else:
                    ref_ids += [i + 1 for i in range(len(mul_ref_image_paths))]

            common_num = min(len(mul_ref_imgs), len(ref_ids))
            mul_ref_imgs = mul_ref_imgs[:common_num]
            ref_ids = ref_ids[:common_num]
        elif (ref_image_path is None and mul_ref_image_paths is None) and args.gen_mode == "ref":
            continue

        # (Optional) read objects and scene images
        if "seg_object_image_path" in validation_df[idx].keys():
            if "ref_ids" in validation_df[idx].keys():
                common_num = min(len(validation_df[idx]["seg_object_image_path"]), len(validation_df[idx]["ref_ids"]["object"]))
                mul_ref_image_paths += validation_df[idx]["seg_object_image_path"][:common_num]
                mul_ref_imgs += [pad_image(Image.open(path).convert("RGB"), skip=False) for path in validation_df[idx]["seg_object_image_path"][:common_num]]
                ref_ids += [i + 100 for i in validation_df[idx]["ref_ids"]["object"][:common_num]]
            else:
                mul_ref_image_paths += validation_df[idx]["seg_object_image_path"]
                mul_ref_imgs += [pad_image(Image.open(path).convert("RGB"), skip=False) for path in validation_df[idx]["seg_object_image_path"]]
                ref_ids += [i + 101 for i in range(len(validation_df[idx]["seg_object_image_path"]))]

        if "scene_path" in validation_df[idx].keys():
            if "ref_ids" in validation_df[idx].keys():
                common_num = min(len(validation_df[idx]["scene_path"]), len(validation_df[idx]["ref_ids"]["scene"]))
                for path in validation_df[idx]["scene_path"][:common_num]:
                    image = Image.open(path).convert("RGB")
                    image = resize_and_crop_scene_image(image, target_width=SIZE_CONFIGS[args.size][1], target_height=SIZE_CONFIGS[args.size][0])
                    mul_ref_imgs.append(image)
                ref_ids += [i + 200 for i in validation_df[idx]["ref_ids"]["scene"][:common_num]]
            else:
                mul_ref_image_paths += validation_df[idx]["scene_path"]
                for path in validation_df[idx]["scene_path"]:
                    image = Image.open(path).convert("RGB")
                    image = resize_and_crop_scene_image(image, target_width=SIZE_CONFIGS[args.size][1], target_height=SIZE_CONFIGS[args.size][0])
                    mul_ref_imgs.append(image)
                ref_ids += [i + 201 for i in range(len(validation_df[idx]["scene_path"]))]

        # Make sure total number of reference is no more than 6
        mul_ref_imgs = mul_ref_imgs[:6]
        ref_ids = ref_ids[:6]
        mul_ref_image_paths = mul_ref_image_paths[:6]

        input_imgs = {}
        if args.gen_mode == "ff":
            input_imgs["ff_img"] = ff_img
            image_path = ff_image_path
        elif args.gen_mode == "lf":
            input_imgs["lf_img"] = lf_img
            image_path = lf_image_path
        elif args.gen_mode == "ref":
            if mul_ref_image_paths is not None:
                input_imgs["ref_img"] = mul_ref_imgs
                image_path = mul_ref_image_paths[0]
            else:
                input_imgs["ref_img"] = ref_img
                image_path = ref_image_path
        elif args.gen_mode == "nul":
            image_path = ""

        face_masks = generate_face_masks(num_ref=2, target_shape=(SIZE_CONFIGS[args.size][1], SIZE_CONFIGS[args.size][0]))
        face_masks = torch.from_numpy(face_masks)

        if isinstance(raw_crop_bbox, dict):
            crop_bbox = []
            for rid in ref_ids:
                if rid < 100:
                    label = f"PERSON_{rid}"
                elif rid < 200:
                    label = f"OBJECT_{rid - 100}"
                elif rid < 300:
                    label = f"SCENE_{rid - 200}"
                try:
                    crop_bbox.append(raw_crop_bbox[label])
                except Exception:
                    crop_bbox.append([-1, -1, -1, -1])
        else:
            crop_bbox = raw_crop_bbox

        if len(crop_bbox) < len(input_imgs):
            w = SIZE_CONFIGS[args.size][1]
            h = SIZE_CONFIGS[args.size][0]
            crop_bbox += [[0, 0, w, h]] * (len(input_imgs) - len(crop_bbox))
        crop_bbox = torch.tensor(crop_bbox).view(-1, 4).long()

        debug_str = f"""
        percentage: {idx + 1}/{len(validation_df)}
           videoid: {videoid}
            prompt: {prompt}
       comp_prompt: {comprehensive_prompt}
  num_of_reference: {len(mul_ref_imgs) if args.gen_mode == 'ref' else 0}
     reference_ids: {ref_ids}
         crop_bbox: {crop_bbox}
        image_path: {image_path}
          gen_mode: {args.gen_mode}
              seed: {base_seed}
               fps: {cfg.sample_fps}
       infer_steps: {args.sample_steps}
     target_height: {SIZE_CONFIGS[args.size][0]}
      target_width: {SIZE_CONFIGS[args.size][1]}
     target_length: {args.frame_num}
      shift_offset: {args.sample_shift}
          boundary: {args.boundary}
          save_dir: {args.save_dir}/{videoid}.mp4
         slg_layer: {args.slg}
            """
        logging.info(debug_str)
        if rank == 0:
            print(debug_str)

        specify_size = True if args.gen_mode in ("ref", "nul") else False

        video = wan_t2v.generate_complex(
            prompt,
            comprehensive_prompt,
            input_imgs,
            ref_ids=torch.tensor(ref_ids).view(1, -1),
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale_text=tuple(args.guide_scale_text),
            guide_scale_img=tuple(args.guide_scale_img),
            seed=base_seed,
            offload_model=args.offload_model,
            specify_size=specify_size,
            one_step_denoising=False,
            face_masks=face_masks,
            correct_id_reference=correct_id_reference,
            slg=args.slg,
            crop_bbox=crop_bbox)

        if rank == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            logging.info(f"Saving generated video to {args.save_dir}/{videoid}.mp4")
            save_video(
                tensor=video[None],
                save_file=f"{args.save_dir}/{videoid}.mp4",
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))

            try:
                with open(f"{args.save_dir}/{videoid}.txt", "w") as f:
                    f.write(prompt)
            except Exception as e:
                logging.info(f"Failed to dump video prompt | {e}")

            if args.gen_mode == "ff":
                ff_img.save(f"{args.save_dir}/{videoid}.png")
            elif args.gen_mode == "lf":
                lf_img.save(f"{args.save_dir}/{videoid}.png")
            elif args.gen_mode == "ref":
                if mul_ref_image_paths is not None:
                    if len(mul_ref_imgs) >= 2:
                        new_img = merge_images_horizontally(*mul_ref_imgs)
                    else:
                        new_img = mul_ref_imgs[0]
                    new_img.save(f"{args.save_dir}/{videoid}.png")
                else:
                    ref_img.save(f"{args.save_dir}/{videoid}.png")

        del video
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    batch_generate(args)

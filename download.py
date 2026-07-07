#!/usr/bin/env python
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Download the model weights required for Aura inference:

  * Wan-AI/Wan2.2-T2V-A14B   -> base checkpoints (UMT5-XXL text encoder + Wan-VAE + tokenizer),
                                used as `--ckpt_dir`.
  * Qwen/Qwen2.5-VL-3B-Instruct -> vision-language backbone for the meta-query encoder,
                                used as `--vlm_dir`.
  * Camellia997/Aura         -> finetuned Aura high-noise / low-noise expert DiT weights,
                                used as `--high_noise_model_dir` / `--low_noise_model_dir`.

Examples
--------
  # download all three into ./weights
  python download.py

  # custom output directory
  python download.py --output_dir /data/weights

  # only one of them
  python download.py --models wan
  python download.py --models qwen
  python download.py --models aura

  # use a mirror endpoint (e.g. in mainland China) and a faster backend
  HF_ENDPOINT=https://hf-mirror.com python download.py --enable_hf_transfer
"""
import argparse
import os
import sys

# Repositories to fetch.
REPOS = {
    "wan": "Wan-AI/Wan2.2-T2V-A14B",
    "qwen": "Qwen/Qwen2.5-VL-3B-Instruct",
    "aura": "Camellia997/Aura",
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Download Wan2.2-T2V-A14B, Qwen2.5-VL-3B-Instruct and Aura weights to a local directory.")
    parser.add_argument(
        "--output_dir", type=str, default="weights",
        help="Local directory to download the weights into (default: ./weights).")
    parser.add_argument(
        "--models", type=str, default="all", choices=["all", "both", "wan", "qwen", "aura"],
        help="Which weights to download: all (=wan+qwen+aura), both (=wan+qwen), or a single "
             "one of wan/qwen/aura (default: all).")
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="Hugging Face access token (or set the HF_TOKEN environment variable).")
    parser.add_argument(
        "--endpoint", type=str, default=None,
        help="Custom Hugging Face endpoint, e.g. https://hf-mirror.com (sets HF_ENDPOINT).")
    parser.add_argument(
        "--max_workers", type=int, default=8,
        help="Number of parallel download workers (default: 8).")
    parser.add_argument(
        "--enable_hf_transfer", action="store_true", default=False,
        help="Enable the hf_transfer accelerated backend (requires `pip install hf_transfer`).")
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    if args.enable_hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed. Install it with:\n"
            "    pip install huggingface_hub\n"
            "(it is also pulled in automatically by `transformers`).")

    token = args.hf_token or os.environ.get("HF_TOKEN")

    if args.models == "all":
        selected = ["wan", "qwen", "aura"]
    elif args.models == "both":
        selected = ["wan", "qwen"]
    else:
        selected = [args.models]

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    for key in selected:
        repo_id = REPOS[key]
        local_dir = os.path.join(args.output_dir, repo_id.split("/")[-1])
        print(f"\n==> Downloading {repo_id}\n    -> {os.path.abspath(local_dir)}")
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=token,
            max_workers=args.max_workers,
            resume_download=True,
        )
        results[key] = os.path.abspath(path)
        print(f"    done: {results[key]}")

    print("\n" + "=" * 70)
    print("Download complete. Use these paths in jobs/infer_*.sh:")
    if "wan" in results:
        print(f"  CKPT_DIR=\"{results['wan']}\"")
    if "aura" in results:
        aura = results["aura"]
        print(f"  HIGH_NOISE_MODEL_DIR=\"{os.path.join(aura, 'high_noise_model')}\"  # finetuned Aura expert")
        print(f"  LOW_NOISE_MODEL_DIR=\"{os.path.join(aura, 'low_noise_model')}\"   # finetuned Aura expert")
    elif "wan" in results:
        wan = results["wan"]
        print(f"  HIGH_NOISE_MODEL_DIR=\"{os.path.join(wan, 'high_noise_model')}\"  # original Wan expert; use Aura weights for Aura results")
        print(f"  LOW_NOISE_MODEL_DIR=\"{os.path.join(wan, 'low_noise_model')}\"   # original Wan expert; use Aura weights for Aura results")
    if "qwen" in results:
        print(f"  VLM_DIR=\"{results['qwen']}\"")
    print("=" * 70)


if __name__ == "__main__":
    main()

# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
import argparse
import os
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download

# Register custom Qwen layers that add AHN support.
from ahn.transformer.qwen2_ahn import register_customized_qwen2
from ahn.transformer.qwen3_ahn import register_customized_qwen3

register_customized_qwen2()
register_customized_qwen3()


def parse_args():
    p = argparse.ArgumentParser(description="Load base model + AHN weights and run a sample generation.")
    p.add_argument(
        "--base-model",
        type=str,
        required=True,
        help='HF repo or local path to the base model (e.g., "Qwen/Qwen2.5-3B-Instruct").',
    )
    p.add_argument(
        "--ahn-path",
        type=str,
        default=None,
        help='Path to AHN-only weights in .safetensors (e.g., "model.safetensors").',
    )
    p.add_argument(
        "--output-path",
        type=str,
        default=None,
        help='Destination to save the merged model weights (base + AHN)',
    )
    return p.parse_args()


def is_local_dir(s: str) -> bool:
    return os.path.isabs(s) or os.path.isdir(s) or s.startswith((".", "/"))


def str_to_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_ahn_overrides(model, ahn_sd):
    model_sd = model.state_dict()
    overlap_keys = [k for k in ahn_sd.keys() if k in model_sd]

    target_dtype = next(model.parameters()).dtype
    filtered = {k: (ahn_sd[k].to(dtype=target_dtype) if ahn_sd[k].dtype != target_dtype else ahn_sd[k])
                for k in overlap_keys}

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    return {
        "overridden_count": len(filtered),
        "total_ahn_keys": len(ahn_sd),
        "overlap_example": overlap_keys[:5],
        "missing": missing,       # should be []
        "unexpected": unexpected  # should be []
    }


def merge_and_safe(args):
    # Load & patch config
    if is_local_dir(args.ahn_path):
        ahn_path = args.ahn_path
    else:
        ahn_path = snapshot_download(
            repo_id=args.ahn_path,
        )
    config = AutoConfig.from_pretrained(ahn_path)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, config=config, torch_dtype=torch.bfloat16)

    # Load ahn state dict
    ahn_ckpt = os.path.join(ahn_path, "model.safetensors")
    ahn_sd = load_file(ahn_ckpt, device="cpu")

    report = load_ahn_overrides(model, ahn_sd)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.save_pretrained(args.output_path)
    except Exception as e:
        print(f"[WARN] Tokenizer not saved: {e}")
    
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(args.output_path)

    model.config.save_ahn_only = False
    model.save_pretrained(
        args.output_path,
        state_dict=model.state_dict(),
        safe_serialization=True,         # -> .safetensors
        max_shard_size="4GB"             # adjust if you prefer larger/smaller shards
    )
    
    print(f"[INFO] Merged model saved to {args.output_path}")


def main():
    args = parse_args()
    merge_and_safe(args)


if __name__ == "__main__":
    main()
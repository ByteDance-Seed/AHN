# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Register custom Qwen layers that add AHN support.
from ahn.transformer.qwen2_ahn import register_customized_qwen2
from ahn.transformer.qwen3_ahn import register_customized_qwen3

register_customized_qwen2()
register_customized_qwen3()


def parse_args():
    p = argparse.ArgumentParser(description="Load base model + AHN weights and run a sample generation.")
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help='Path to the ahn-augmented model.',
    )
    p.add_argument(
        "--sliding-window",
        type=int,
        default=32640,
        help="Sliding window length for lossless attention memory.",
    )
    p.add_argument(
        "--num-attention-sink",
        type=int,
        default=128,
        help="Number of attention sink tokens used as anchors.",
    )
    # Optional extras you might want to tweak quickly:
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for loading and inference.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max new tokens for the demo generation.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Write a 10,000-word poem.",
        help="Prompt for the demo generation.",
    )
    return p.parse_args()


def str_to_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    args = parse_args()
    dtype = str_to_dtype(args.dtype)

    if not torch.cuda.is_available():
        raise ValueError("GPU is not available")
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None
    ).to(device)

    model.eval()

    inputs = tokenizer(
        args.prompt,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        context_length = inputs.input_ids.shape[-1]
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            do_sample=False,
        )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

    print(pred)


if __name__ == "__main__":
    main()

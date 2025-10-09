# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

def guess_index_file(model_dir: Path):
    candidates = list(model_dir.glob("*.index.json"))
    if not candidates:
        raise FileNotFoundError("No *.index.json found in model_dir.")
    if len(candidates) > 1:
        # Prefer safetensors index if both exist
        for c in candidates:
            if c.name.endswith(".safetensors.index.json"):
                return c
    return candidates[0]

def rename_key(k: str, old: str, new: str):
    return k.replace(old, new)

def process_safetensors_shard(src: Path, dst: Path, old: str, new: str, dry: bool):
    from safetensors.torch import load_file, save_file
    if dry:
        # Just list what would change (without loading tensors)
        # We can peek keys via safe_open (no tensor data load)
        from safetensors import safe_open
        with safe_open(str(src), framework="pt", device="cpu") as f:
            for k in f.keys():
                new_k = rename_key(k, old, new)
                if new_k != k:
                    print(f"[DRY] {src.name}: {k} -> {new_k}")
        return

    sd = load_file(str(src))  # loads this shard only
    new_sd = {}
    changed = 0
    for k, v in sd.items():
        new_k = rename_key(k, old, new)
        if new_k != k:
            changed += 1
        new_sd[new_k] = v
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(new_sd, str(dst))
    print(f"[OK] {src.name} -> {dst.name}  (renamed {changed} keys)")

def process_pt_shard(src: Path, dst: Path, old: str, new: str, dry: bool):
    import torch
    if dry:
        # Need to load state_dict to see keys for .bin
        sd = torch.load(str(src), map_location="cpu")
        for k in sd.keys():
            new_k = rename_key(k, old, new)
            if new_k != k:
                print(f"[DRY] {src.name}: {k} -> {new_k}")
        return

    import torch
    sd = torch.load(str(src), map_location="cpu")
    new_sd = {}
    changed = 0
    for k, v in sd.items():
        new_k = rename_key(k, old, new)
        if new_k != k:
            changed += 1
        new_sd[new_k] = v
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_sd, str(dst))
    print(f"[OK] {src.name} -> {dst.name}  (renamed {changed} keys)")

def main():
    ap = argparse.ArgumentParser(description="Rename module prefix in sharded HF checkpoint.")
    ap.add_argument("--model-dir", required=True, help="Path to directory containing shards + index json.")
    ap.add_argument("--out-dir", required=True, help="Where to write renamed checkpoint.")
    ap.add_argument("--old", required=True, help="Substring to replace in parameter keys (e.g., 'old_module').")
    ap.add_argument("--new", required=True, help="Replacement substring (e.g., 'new_module').")
    ap.add_argument("--dry-run", action="store_true", help="Print what would change without writing files.")
    ap.add_argument("--copy-metadata", action="store_true",
                    help="Copy config.json, tokenizer files, etc., to out-dir.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_file = guess_index_file(model_dir)
    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    # Index schema: {"metadata": {...}, "weight_map": {"param.name": "shard_file"}}
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Unexpected index format in: {index_file}")

    # 1) Rename keys in index weight_map
    new_weight_map = {}
    changes = 0
    for k, shard in weight_map.items():
        new_k = rename_key(k, args.old, args.new)
        if new_k != k:
            changes += 1
        new_weight_map[new_k] = shard
    index["weight_map"] = new_weight_map
    print(f"[INFO] Will rename {changes} keys in index mapping.")

    # 2) Process each shard file listed in weight_map
    shard_files = sorted(set(weight_map.values()))
    print(f"[INFO] Found {len(shard_files)} shard files.")

    for shard_name in shard_files:
        src = model_dir / shard_name
        dst = out_dir / shard_name
        if not src.exists():
            raise FileNotFoundError(f"Shard listed in index not found: {src}")

        if shard_name.endswith(".safetensors"):
            process_safetensors_shard(src, dst, args.old, args.new, args.dry_run)
        elif shard_name.endswith(".bin"):
            process_pt_shard(src, dst, args.old, args.new, args.dry_run)
        else:
            raise ValueError(f"Unknown shard extension: {shard_name}")

    # 3) Write updated index
    new_index_name = index_file.name
    new_index_path = out_dir / new_index_name
    if args.dry_run:
        print(f"[DRY] Would write updated index to: {new_index_path}")
    else:
        with open(new_index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote updated index: {new_index_path}")

    # 4) Optionally copy non-weight metadata files
    if args.copy_metadata:
        for fname in [
            "config.json",
            "generation_config.json",
        ] + [p.name for p in model_dir.glob("tokenizer*")] + [p.name for p in model_dir.glob("special_tokens*")]:
            src = model_dir / fname
            if src.exists():
                dst = out_dir / fname
                if args.dry_run:
                    print(f"[DRY] Would copy {src.name} -> {dst}")
                else:
                    shutil.copy2(src, dst)

    print("[DONE]")

if __name__ == "__main__":
    main()

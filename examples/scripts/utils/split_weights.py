#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable

import torch
from safetensors.torch import safe_open, save_file


def find_weight_files(folder: Path) -> Iterable[Path]:
    """
    Return model weight files saved by HF `save_pretrained`.
    Supports:
      - *.safetensors (sharded or single)
      - pytorch_model*.bin (sharded or single)
    """
    # Prefer safetensors if both exist
    st_files = sorted(folder.glob("*.safetensors"))
    if st_files:
        return st_files

    # Fallback to PyTorch .bin
    bin_files = sorted(folder.glob("pytorch_model*.bin"))
    if bin_files:
        return bin_files

    # Nothing found
    return []


def read_index_json(folder: Path) -> Dict[str, str]:
    """
    If an index file exists (for sharded checkpoints), return {param_key: shard_file}.
    """
    idx = {}
    for name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        p = folder / name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # format: {"weight_map": {"xxx": "pytorch_model-00001-of-000xx.bin", ...}}
            idx = data.get("weight_map", {})
            break
    return idx


def collect_ahn_from_safetensors(files: Iterable[Path], key_filter, index_map: Dict[str, str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if index_map:
        # Only open shards that contain ahn-keys (faster on huge models)
        shards_needed = set()
        for k, shard in index_map.items():
            if key_filter(k):
                shards_needed.add(shard)
        shard_set = {f.name for f in files}
        target_files = [f for f in files if f.name in shards_needed and f.name in shard_set]
    else:
        target_files = list(files)

    for f in target_files:
        with safe_open(f.as_posix(), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if key_filter(k):
                    out[k] = sf.get_tensor(k)  # already on CPU
    return out


def collect_ahn_from_bin(files: Iterable[Path], key_filter, index_map: Dict[str, str]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if index_map:
        shards_needed = set()
        for k, shard in index_map.items():
            if key_filter(k):
                shards_needed.add(shard)
        files = [f for f in files if f.name in shards_needed]

    for f in files:
        # map_location='cpu' to avoid GPU memory usage
        state = torch.load(f.as_posix(), map_location="cpu")
        # state can be plain dict or {"state_dict": ...}
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        for k, v in state.items():
            if key_filter(k):
                out[k] = v.detach().cpu()
        # release ASAP
        del state
    return out


def main():
    ap = argparse.ArgumentParser(description="Extract all parameters whose names contain 'ahn' into model.safetensors.")
    ap.add_argument("folder", type=Path, help="Path to folder saved by Hugging Face save_pretrained (contains config.json, weights, etc.)")
    ap.add_argument("--out", type=Path, default=None, help="Output .safetensors path (default: <folder>/model.safetensors)")
    ap.add_argument("--substring", type=str, default="ahn", help="Substring to match in parameter names (default: 'ahn')")
    ap.add_argument("--case-sensitive", action="store_true", help="Make the match case-sensitive")
    args = ap.parse_args()

    folder: Path = args.folder
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    out_path = args.out or (folder / "model.safetensors")

    if args.case_sensitive:
        key_filter = lambda k: args.substring in k
    else:
        sub = args.substring.lower()
        key_filter = lambda k: sub in k.lower()

    files = list(find_weight_files(folder))
    if not files:
        raise SystemExit(
            f"No model weight files found in {folder}. Expected *.safetensors or pytorch_model*.bin"
        )

    index_map = read_index_json(folder)

    if files[0].suffix == ".safetensors":
        ahn_sd = collect_ahn_from_safetensors(files, key_filter, index_map)
    else:
        ahn_sd = collect_ahn_from_bin(files, key_filter, index_map)

    if not ahn_sd:
        raise SystemExit(
            f"No parameters matched substring '{args.substring}'. "
            "Try adjusting --substring or --case-sensitive."
        )

    # Save in a single consolidated safetensors file
    save_file(ahn_sd, out_path.as_posix())
    total = sum(v.numel() for v in ahn_sd.values())
    print(f"[OK] Saved {len(ahn_sd)} tensors ({total:,} params) to {out_path}")


if __name__ == "__main__":
    main()

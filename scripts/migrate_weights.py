#!/usr/bin/env python3
"""Migrate pre-trained weights from MinkowskiEngine to SpConv v2.x format.

Handles:
1. Nested checkpoint format: ckpt['models']['latest'] → state dict
2. Kernel shape transformation: ME (K^3, C_in, C_out) → SpConv (C_out, kD, kH, kW, C_in)
3. Key renaming: *.kernel → *.conv.weight
4. 1x1 convolution kernels: ME (C_in, C_out) → SpConv (C_out, 1, 1, 1, C_in)

Usage:
    python scripts/migrate_weights.py \\
        --input model_file/PointGroup-PAPER.pt \\
        --output model_file/PointGroup-PAPER-spconv.pt \\
        [--model-key latest] [--verify] [--dry-run]
"""

import argparse
import math
import sys
from collections import OrderedDict

import torch


def reshape_kernel(tensor):
    """Reshape an ME kernel tensor to SpConv weight format.

    ME stores kernels as:
      - 3D: (K^3, C_in, C_out) for kernel_size > 1
      - 2D: (C_in, C_out) for kernel_size = 1

    SpConv stores weights as:
      - 5D: (C_out, kD, kH, kW, C_in)
    """
    if tensor.dim() == 2:
        # kernel_size=1: (C_in, C_out) → (C_out, 1, 1, 1, C_in)
        c_in, c_out = tensor.shape
        return tensor.t().reshape(c_out, 1, 1, 1, c_in)

    elif tensor.dim() == 3:
        # kernel_size>1: (K^3, C_in, C_out) → (C_out, K, K, K, C_in)
        k3, c_in, c_out = tensor.shape
        k = round(k3 ** (1.0 / 3.0))
        assert k * k * k == k3, f"Kernel size {k3} is not a perfect cube"
        return tensor.reshape(k, k, k, c_in, c_out).permute(4, 0, 1, 2, 3).contiguous()

    else:
        raise ValueError(f"Unexpected kernel shape: {tensor.shape}")


def migrate_state_dict(state_dict):
    """Convert all keys and kernel shapes from ME to SpConv format."""
    new_sd = OrderedDict()
    changes = []

    for key, value in state_dict.items():
        if key.endswith(".kernel"):
            # Rename key and reshape kernel
            new_key = key[:-len(".kernel")] + ".conv.weight"
            new_value = reshape_kernel(value)
            changes.append((key, new_key, tuple(value.shape), tuple(new_value.shape)))
            new_sd[new_key] = new_value
        else:
            # Copy as-is (BatchNorm, MLP heads, etc.)
            new_sd[key] = value

    return new_sd, changes


def extract_state_dict(checkpoint, model_key="latest"):
    """Extract state dict from nested checkpoint format."""
    if not isinstance(checkpoint, dict):
        return checkpoint, "flat"

    # Format 1: ckpt['models'][model_key] (SegmentAnyTree format)
    if "models" in checkpoint:
        models = checkpoint["models"]
        if model_key in models:
            return models[model_key], f"models.{model_key}"
        # Try first available key
        first_key = list(models.keys())[0]
        print(f"Warning: '{model_key}' not found, using '{first_key}'")
        return models[first_key], f"models.{first_key}"

    # Format 2: ckpt['model_state_dict']
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], "model_state_dict"

    # Format 3: ckpt['state_dict']
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"], "state_dict"

    # Assume the dict itself is the state dict
    return checkpoint, "root"


def main():
    parser = argparse.ArgumentParser(
        description="Migrate MinkowskiEngine weights to SpConv format"
    )
    parser.add_argument("--input", required=True, help="Path to ME checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Path to save SpConv checkpoint (.pt)")
    parser.add_argument("--model-key", default="latest",
                        help="Key in ckpt['models'] to use (default: latest)")
    parser.add_argument("--verify", action="store_true", help="Print full key listing")
    parser.add_argument("--dry-run", action="store_true", help="Don't save, just show changes")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")

    state_dict, location = extract_state_dict(checkpoint, args.model_key)
    print(f"State dict from: {location} ({len(state_dict)} parameters)")

    # Migrate
    new_sd, changes = migrate_state_dict(state_dict)

    # Report
    print(f"\nConverted {len(changes)} kernel parameters:")
    for old_key, new_key, old_shape, new_shape in changes:
        print(f"  {old_key} {old_shape} -> {new_key} {new_shape}")

    unchanged = len(state_dict) - len(changes)
    print(f"\n{unchanged} parameters unchanged (BatchNorm, MLP heads, etc.)")
    print(f"Total: {len(new_sd)} parameters")

    if args.verify:
        print("\n=== Full key listing ===")
        for key, value in new_sd.items():
            print(f"  {key}: {tuple(value.shape)}")

    if not args.dry_run:
        # Rebuild checkpoint preserving metadata
        if isinstance(checkpoint, dict) and "models" in checkpoint:
            new_ckpt = dict(checkpoint)
            new_ckpt["models"] = dict(checkpoint["models"])
            # Convert all model variants
            for mk in new_ckpt["models"]:
                sd = new_ckpt["models"][mk]
                if isinstance(sd, dict) and any(k.endswith(".kernel") for k in sd):
                    converted, _ = migrate_state_dict(sd)
                    new_ckpt["models"][mk] = converted
                    print(f"  Converted models.{mk}")
            torch.save(new_ckpt, args.output)
        else:
            torch.save(new_sd, args.output)

        print(f"\nSaved: {args.output}")
    else:
        print("\n[dry-run] No file saved")


if __name__ == "__main__":
    main()

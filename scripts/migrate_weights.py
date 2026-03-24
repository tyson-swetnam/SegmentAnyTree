#!/usr/bin/env python3
"""Migrate pre-trained weights from MinkowskiEngine format to SparseConv3d/SpConv format.

The key differences between ME and SparseConv3d state dicts:

1. Convolution weights:
   - ME api_modules: stored as `.kernel` (parameter name in ME.MinkowskiConvolution)
   - SparseConv3d modules: stored as `.conv.weight` (SpConv conv layers use .weight,
     wrapped in our Conv3d.conv attribute)

2. BatchNorm weights:
   - ME api_modules: ME.MinkowskiBatchNorm stores as `.bn.weight`, `.bn.bias`, etc.
     In the ME ResNetDown, BatchNorm is used directly, so keys are like:
     `conv_in.1.bn.weight` (where .1 is the BatchNorm in the Seq)
   - SparseConv3d modules: snn.BatchNorm wraps nn.BatchNorm1d in `.bn`, so keys are:
     `conv_in.1.bn.weight` (same! because our SpConv BatchNorm uses self.bn = nn.BatchNorm1d)

3. Module hierarchy:
   - Both use UnwrappedUnetBasedModel with down_modules/up_modules
   - Both use ResBlock/ResNetDown/ResNetUp with conv_in and blocks
   - The ResBlock internal structure differs:
     ME: block uses ME.MinkowskiConvolution directly (kernel param), ME.MinkowskiBatchNorm (.bn submodule)
     SparseConv3d: block uses snn.Conv3d (wraps SpConv in .conv), snn.BatchNorm (wraps BN1d in .bn)

Usage:
    python scripts/migrate_weights.py \\
        --input model_file/PointGroup-PAPER.pt \\
        --output model_file/PointGroup-PAPER-spconv.pt \\
        [--verify] [--dry-run]
"""

import argparse
import re
import sys
import torch
from collections import OrderedDict


def remap_key(key: str) -> str:
    """Remap a MinkowskiEngine state dict key to SparseConv3d/SpConv format.

    ME api_modules structure (ResNetDown):
        conv_in.0 = ME.MinkowskiConvolution  -> has .kernel parameter
        conv_in.1 = ME.MinkowskiBatchNorm    -> has .bn.weight, .bn.bias, .bn.running_mean, etc.
        conv_in.2 = ME.MinkowskiReLU         -> no parameters
        blocks.N.block.0 = ME.MinkowskiConvolution -> .kernel
        blocks.N.block.1 = ME.MinkowskiBatchNorm -> .bn.weight, ...
        blocks.N.downsample.0 = ME.MinkowskiConvolution -> .kernel
        blocks.N.downsample.1 = ME.MinkowskiBatchNorm -> .bn.weight, ...

    SparseConv3d modules structure (ResNetDown):
        conv_in.0 = snn.Conv3d  -> has .conv.weight parameter
        conv_in.1 = snn.BatchNorm -> has .bn.weight, .bn.bias, .bn.running_mean, etc.
        conv_in.2 = snn.ReLU -> no parameters
        blocks.N.block.0 = snn.Conv3d (passed as `convolution` arg) -> .conv.weight
        blocks.N.block.1 = snn.BatchNorm -> .bn.weight, ...
        blocks.N.downsample.0 = snn.Conv3d -> .conv.weight
        blocks.N.downsample.1 = snn.BatchNorm -> .bn.weight, ...

    Key differences:
        - ME convolution `.kernel` -> SpConv `.conv.weight`
        - BatchNorm keys stay the same (.bn.weight, .bn.bias, etc.)
    """
    # Convolution kernel: ME uses `.kernel`, SpConv wrapper uses `.conv.weight`
    # Pattern: any path ending in `.kernel` where the parent is a convolution module
    # Examples:
    #   Backbone.down_modules.0.conv_in.0.kernel -> Backbone.down_modules.0.conv_in.0.conv.weight
    #   Backbone.up_modules.0.conv_in.0.kernel -> Backbone.up_modules.0.conv_in.0.conv.weight
    #   ScorerUnet.down_modules.0.blocks.0.block.0.kernel -> ...conv.weight
    if key.endswith(".kernel"):
        return key[:-len(".kernel")] + ".conv.weight"

    # ME.MinkowskiBatchNorm has internal .bn module, so keys are already like:
    #   .bn.weight, .bn.bias, .bn.running_mean, .bn.running_var, .bn.num_batches_tracked
    # Our SpConv BatchNorm also stores weights in .bn (nn.BatchNorm1d), so these match.

    return key


def migrate_state_dict(state_dict: dict) -> OrderedDict:
    """Remap all keys in a state dict from ME format to SparseConv3d/SpConv format."""
    new_state_dict = OrderedDict()
    changes = []

    for old_key, value in state_dict.items():
        new_key = remap_key(old_key)
        new_state_dict[new_key] = value
        if new_key != old_key:
            changes.append((old_key, new_key, tuple(value.shape)))

    return new_state_dict, changes


def main():
    parser = argparse.ArgumentParser(description="Migrate ME weights to SpConv format")
    parser.add_argument("--input", required=True, help="Path to ME checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Path to save SpConv checkpoint (.pt)")
    parser.add_argument("--verify", action="store_true", help="Print detailed key mapping")
    parser.add_argument("--dry-run", action="store_true", help="Don't save, just show changes")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.input}")
    checkpoint = torch.load(args.input, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        is_wrapped = True
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        is_wrapped = True
    elif isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
        is_wrapped = False
    else:
        print(f"Unexpected checkpoint format: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"Keys: {list(checkpoint.keys())[:20]}")
        sys.exit(1)

    print(f"Found {len(state_dict)} parameters")

    new_state_dict, changes = migrate_state_dict(state_dict)

    print(f"\nRemapped {len(changes)} keys:")
    for old_key, new_key, shape in changes:
        print(f"  {old_key} -> {new_key}  {shape}")

    unchanged = len(state_dict) - len(changes)
    print(f"\n{unchanged} keys unchanged")

    if args.verify:
        print("\n--- Full key listing ---")
        for key, value in new_state_dict.items():
            print(f"  {key}: {tuple(value.shape)}")

    if not args.dry_run:
        if is_wrapped:
            # Preserve the original checkpoint structure
            new_checkpoint = dict(checkpoint)
            if "model_state_dict" in checkpoint:
                new_checkpoint["model_state_dict"] = new_state_dict
            else:
                new_checkpoint["state_dict"] = new_state_dict
            torch.save(new_checkpoint, args.output)
        else:
            torch.save(new_state_dict, args.output)
        print(f"\nSaved migrated checkpoint to: {args.output}")
    else:
        print("\n[dry-run] No file saved")


if __name__ == "__main__":
    main()

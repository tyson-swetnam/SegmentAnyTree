"""Cache management for the inference pipeline."""

import argparse
import os
import yaml


def clear_inference_cache(eval_yaml_path):
    """Clear cached dataset files to avoid stale data between inference runs.

    Reads the checkpoint_dir from eval.yaml, finds the data.dataroot from
    the checkpoint's hydra overrides, and removes processed files.
    """
    with open(eval_yaml_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    checkpoint_dir = data.get('checkpoint_dir', '')
    overrides_path = os.path.join(checkpoint_dir, '.hydra/overrides.yaml')

    if not os.path.exists(overrides_path):
        print(f"No overrides file at {overrides_path}, skipping cache clear")
        return

    with open(overrides_path) as f:
        overrides = yaml.load(f, Loader=yaml.FullLoader)

    dataroot = None
    for item in overrides:
        if 'data.dataroot' in item:
            dataroot = item.split('=')[1].strip()

    if not dataroot:
        print("Could not find data.dataroot in overrides, skipping cache clear")
        return

    cache_path = os.path.join(dataroot, 'treeinsfused', 'processed_0.2_test')
    print(f"Clearing cache: {cache_path}")

    if os.path.exists(cache_path):
        for filename in os.listdir(cache_path):
            filepath = os.path.join(cache_path, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
                print(f"Removed: {filepath}")
    else:
        print(f"Cache path does not exist: {cache_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clear inference cache.')
    parser.add_argument('--eval_yaml', type=str, required=True)
    args = parser.parse_args()
    clear_inference_cache(args.eval_yaml)

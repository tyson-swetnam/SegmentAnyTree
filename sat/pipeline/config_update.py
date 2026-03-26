"""Dynamic eval.yaml configuration for inference runs."""

import os
import argparse
import yaml
from collections import OrderedDict


def get_all_ply_paths(directory):
    """Recursively find all .ply files in directory."""
    ply_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.ply'):
                ply_paths.append(os.path.join(root, filename))
    return ply_paths


def modify_eval_yaml(yaml_path, ply_folder, output_dir=None):
    """Update eval.yaml with PLY file paths and output directory.

    Args:
        yaml_path: Path to eval.yaml to modify.
        ply_folder: Folder containing .ply files to add to data.fold.
        output_dir: Optional output directory for hydra.run.dir.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    data = OrderedDict(data)
    ply_paths = get_all_ply_paths(ply_folder)
    if not ply_paths:
        raise FileNotFoundError(f"No .ply files found in: {ply_folder}")

    data['data']['fold'] = list(ply_paths)
    if output_dir:
        data['hydra']['run']['dir'] = output_dir

    # Resolve checkpoint_dir to absolute path so it works regardless of
    # Hydra's working directory change during eval.py execution.
    sat_root = os.environ.get('SAT_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ckpt_dir = data.get('checkpoint_dir', 'model_file')
    if not os.path.isabs(ckpt_dir):
        data['checkpoint_dir'] = os.path.join(sat_root, ckpt_dir)

    def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
        class OrderedDumper(Dumper):
            pass
        def _dict_representer(dumper, data):
            return dumper.represent_mapping(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())
        OrderedDumper.add_representer(OrderedDict, _dict_representer)
        return yaml.dump(data, stream, OrderedDumper, **kwds)

    with open(yaml_path, 'w') as f:
        ordered_dump(data, f, Dumper=yaml.SafeDumper, default_flow_style=False)

    print(f"Updated {yaml_path} with {len(ply_paths)} files")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update eval.yaml with PLY file paths.')
    parser.add_argument('yaml_file_path', help='Path to eval.yaml')
    parser.add_argument('folder_path', help='Folder containing .ply files')
    parser.add_argument('output_dir_path', help='Output directory')
    args = parser.parse_args()
    modify_eval_yaml(args.yaml_file_path, args.folder_path, args.output_dir_path)

"""Rename inference output files from generic indices to descriptive names."""

import os
import sys
import yaml


def rename_instance_results(yaml_file, directory):
    """Rename result_N.ply -> instance_segmentation_<original_name>.ply"""
    with open(yaml_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    fold_section = data.get('data', {}).get('fold', [])
    for index, file_path in enumerate(fold_section):
        file_name = os.path.basename(file_path)
        old_path = os.path.join(directory, f'result_{index}.ply')
        new_path = os.path.join(directory, f'instance_segmentation_{file_name}')

        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f'Renamed {old_path} to {new_path}')
        else:
            print(f'Warning: {old_path} not found')


def rename_segmentation_results(yaml_file, directory):
    """Rename semantic_result_N.ply -> semantic_segmentation_<original_name>.ply"""
    with open(yaml_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    fold_section = data.get('data', {}).get('fold', [])
    for index, file_path in enumerate(fold_section):
        file_name = os.path.basename(file_path)
        old_path = os.path.join(directory, f'semantic_result_{index}.ply')
        new_path = os.path.join(directory, f'semantic_segmentation_{file_name}')

        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f'Renamed {old_path} to {new_path}')
        else:
            print(f'Warning: {old_path} not found')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python result_rename.py <instance|semantic> <yaml_file> <directory>')
        sys.exit(1)

    mode, yaml_file, directory = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == 'instance':
        rename_instance_results(yaml_file, directory)
    elif mode == 'semantic':
        rename_segmentation_results(yaml_file, directory)
    else:
        print(f'Unknown mode: {mode}. Use "instance" or "semantic".')
        sys.exit(1)

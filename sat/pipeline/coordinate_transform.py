"""UTM <-> local coordinate transformations for point clouds.

The model expects points in a local coordinate system (near origin).
This module subtracts the minimum x/y/z values before inference and
restores them afterward. Min values are saved to JSON for round-tripping.
"""

import argparse
import json
import os
from pathlib import Path

from joblib import Parallel, delayed

from sat.io.las_io import las_to_pandas
from sat.io.ply_io import ply_to_pandas, pandas_to_ply


def utm_to_local(input_file_path, output_file_path, json_file_path):
    """Transform a point cloud from UTM to local coordinates.

    Subtracts minimum x, y, z values and saves them to a JSON file
    so coordinates can be restored later.
    """
    is_ply = input_file_path.endswith('.ply')
    coord_names = ['x', 'y', 'z'] if is_ply else ['X', 'Y', 'Z']
    print(f"Processing UTM->local: {input_file_path}")

    points_df = ply_to_pandas(input_file_path) if is_ply else las_to_pandas(input_file_path)

    min_values = {name: points_df[name].min() for name in coord_names}
    for name in coord_names:
        points_df[name] -= min_values[name]

    min_values_list = [float(val) for val in min_values.values()]
    with open(json_file_path, 'w') as f:
        json.dump(min_values_list, f)

    pandas_to_ply(points_df, output_file_path)


def local_to_utm(input_file_path, json_file_path, output_file_path):
    """Restore UTM coordinates from local coordinates using saved min values."""
    is_ply = input_file_path.endswith('.ply')
    coord_names = ['x', 'y', 'z'] if is_ply else ['X', 'Y', 'Z']

    points_df = ply_to_pandas(input_file_path) if is_ply else las_to_pandas(input_file_path)

    with open(json_file_path, 'r') as f:
        min_values = json.load(f)

    for name, min_val in zip(coord_names, min_values):
        points_df[name] = points_df[name].astype(float) + min_val

    pandas_to_ply(points_df, output_file_path)


def _process_file(filename, input_folder, output_folder):
    """Process a single file for UTM->local transform."""
    if not filename.endswith(('.ply', '.las', '.laz')):
        return
    input_path = os.path.join(input_folder, filename)
    base = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, f"{base}_out.ply")
    json_path = os.path.join(output_folder, f"{base}_out_min_values.json")
    utm_to_local(input_path, output_path, json_path)


def utm_to_local_folder(input_folder, output_folder, n_jobs=4):
    """Transform all point clouds in a folder from UTM to local coordinates."""
    os.makedirs(output_folder, exist_ok=True)
    filenames = os.listdir(input_folder)
    print(f"Processing {len(filenames)} files...")
    Parallel(n_jobs=n_jobs)(
        delayed(_process_file)(f, input_folder, output_folder) for f in filenames
    )
    print(f"Output files saved in: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform point cloud coordinates UTM <-> local.')
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('--reverse', action='store_true', help='Reverse: local->UTM')
    parser.add_argument('-j', '--jobs', type=int, default=4)
    args = parser.parse_args()

    if args.reverse:
        raise NotImplementedError("Batch local->UTM not yet implemented. Use local_to_utm() directly.")
    else:
        utm_to_local_folder(args.input_folder, args.output_folder, n_jobs=args.jobs)

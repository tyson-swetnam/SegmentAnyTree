"""UTM <-> local coordinate transformations for point clouds.

The model expects points in a local coordinate system (near origin).
This module subtracts the minimum x/y/z values before inference and
restores them afterward. Min values are saved to JSON for round-tripping.

Optimized for large point clouds (100M+ points):
- Uses numpy arrays directly instead of pandas DataFrames
- Avoids row-by-row iteration in PLY writing
- Coordinate subtraction is vectorized numpy (multi-threaded via BLAS)
"""

import argparse
import json
import os
import time

import numpy as np
from joblib import Parallel, delayed

from sat.io.las_io import las_to_numpy
from sat.io.ply_io import ply_to_numpy, numpy_to_ply


def utm_to_local(input_file_path, output_file_path, json_file_path):
    """Transform a point cloud from UTM to local coordinates.

    Uses numpy arrays directly — no pandas DataFrames — for maximum speed
    on large files. Coordinate subtraction is vectorized.
    """
    t0 = time.time()
    is_ply = input_file_path.endswith('.ply')
    print(f"[UTM->local] Reading: {os.path.basename(input_file_path)}")

    if is_ply:
        structured, prop_names = ply_to_numpy(input_file_path)
        coord_names = ['x', 'y', 'z']
        # Convert structured array to dict of arrays for mutation
        arrays = {name: np.array(structured[name], dtype=np.float64) for name in prop_names}
    else:
        arrays, prop_names, _ = las_to_numpy(input_file_path)
        coord_names = ['X', 'Y', 'Z']
        # Ensure float64 for coordinate arithmetic
        for c in coord_names:
            arrays[c] = arrays[c].astype(np.float64)

    n_points = len(arrays[coord_names[0]])
    t_read = time.time() - t0
    print(f"[UTM->local] Read {n_points:,} points in {t_read:.1f}s")

    # Compute and subtract min values (vectorized numpy, uses all cores via BLAS)
    t1 = time.time()
    min_values = [float(arrays[c].min()) for c in coord_names]
    for c, mv in zip(coord_names, min_values):
        arrays[c] -= mv  # in-place subtract, no copy

    t_transform = time.time() - t1
    print(f"[UTM->local] Transform in {t_transform:.1f}s, min_values={min_values}")

    # Save min values for later restoration
    with open(json_file_path, 'w') as f:
        json.dump(min_values, f)

    # Write PLY output using numpy path (no pandas, no row iteration)
    t2 = time.time()
    # Output always uses lowercase coordinate names for PLY
    output_arrays = {}
    out_names = []
    for name in prop_names:
        if name in coord_names:
            # Map LAS uppercase to PLY lowercase
            out_name = name.lower()
            output_arrays[out_name] = arrays[name].astype(np.float32)
        else:
            out_name = name
            output_arrays[out_name] = np.asarray(arrays[name], dtype=np.float32)
        out_names.append(out_name)

    numpy_to_ply(output_arrays, out_names, output_file_path)
    t_write = time.time() - t2
    t_total = time.time() - t0
    print(f"[UTM->local] Wrote PLY in {t_write:.1f}s (total: {t_total:.1f}s)")


def local_to_utm(input_file_path, json_file_path, output_file_path):
    """Restore UTM coordinates from local coordinates using saved min values."""
    is_ply = input_file_path.endswith('.ply')
    coord_names = ['x', 'y', 'z'] if is_ply else ['X', 'Y', 'Z']

    if is_ply:
        structured, prop_names = ply_to_numpy(input_file_path)
        arrays = {name: np.array(structured[name], dtype=np.float64) for name in prop_names}
    else:
        arrays, prop_names, _ = las_to_numpy(input_file_path)
        for c in coord_names:
            arrays[c] = arrays[c].astype(np.float64)

    with open(json_file_path, 'r') as f:
        min_values = json.load(f)

    for c, mv in zip(coord_names, min_values):
        arrays[c] += mv

    # Write back as PLY
    out_names = [c.lower() if c in coord_names else c for c in prop_names]
    output_arrays = {}
    for name, out_name in zip(prop_names, out_names):
        output_arrays[out_name] = arrays[name].astype(np.float32)
    numpy_to_ply(output_arrays, out_names, output_file_path)


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
    """Transform all point clouds in a folder from UTM to local coordinates.

    Files are processed in parallel using joblib. Within each file,
    numpy operations are vectorized across all available cores.
    """
    os.makedirs(output_folder, exist_ok=True)
    filenames = [f for f in os.listdir(input_folder)
                 if f.endswith(('.ply', '.las', '.laz'))]
    print(f"Processing {len(filenames)} files (n_jobs={n_jobs})...")
    if len(filenames) == 1:
        # Single file: run directly (no joblib overhead)
        _process_file(filenames[0], input_folder, output_folder)
    else:
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

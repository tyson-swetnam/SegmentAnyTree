"""COPC octant-parallel inference: split a single large COPC file into
spatial tiles and process each tile independently across GPUs.

Cloud Optimized Point Clouds (COPC) store data in an octree structure where
each node (octant) is independently accessible. This module exploits that
structure to enable single-file multi-GPU inference.

Usage:
    from sat.pipeline.copc_parallel import split_copc_to_tiles, merge_tile_results

    tiles = split_copc_to_tiles("input.copc.laz", "workdir/", num_tiles=4, overlap=2.0)
    # ... run inference on each tile (on separate GPUs) ...
    merge_tile_results("workdir/", "output.laz", overlap=2.0)
"""

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import laspy
import numpy as np
from laspy import Bounds
from laspy.copc import CopcReader

log = logging.getLogger(__name__)


@dataclass
class Tile:
    """A spatial tile extracted from a COPC file."""
    index: int
    bounds: Bounds
    file_path: str
    point_count: int = 0
    # Padded bounds include the overlap buffer
    padded_bounds: Optional[Bounds] = None


def _compute_grid(header, num_tiles: int) -> Tuple[int, int]:
    """Compute a 2D grid (nx, ny) that's close to sqrt(num_tiles) and covers the area."""
    x_range = header.x_max - header.x_min
    y_range = header.y_max - header.y_min
    aspect = x_range / max(y_range, 1e-6)

    ny = max(1, round(math.sqrt(num_tiles / max(aspect, 1e-6))))
    nx = max(1, math.ceil(num_tiles / ny))

    # Adjust to not exceed num_tiles
    while nx * ny > num_tiles and ny > 1:
        ny -= 1
    while nx * ny < num_tiles:
        nx += 1

    return nx, ny


def split_copc_to_tiles(
    copc_path: str,
    work_dir: str,
    num_tiles: int = 4,
    overlap: float = 2.0,
) -> List[Tile]:
    """Split a COPC file into spatial tiles for parallel processing.

    Parameters
    ----------
    copc_path : str
        Path to input .copc.laz file.
    work_dir : str
        Directory to write tile files into.
    num_tiles : int
        Number of tiles (typically = number of GPUs).
    overlap : float
        Overlap buffer in meters at tile boundaries. Trees spanning tile edges
        need context from neighboring tiles to be segmented correctly.

    Returns
    -------
    List[Tile]
        List of tiles with file paths and bounds metadata.
    """
    os.makedirs(work_dir, exist_ok=True)

    with CopcReader.open(copc_path) as reader:
        header = reader.header
        total_points = header.point_count
        log.info(f"COPC file: {total_points:,} points, "
                 f"x=[{header.x_min:.1f}, {header.x_max:.1f}], "
                 f"y=[{header.y_min:.1f}, {header.y_max:.1f}]")

        nx, ny = _compute_grid(header, num_tiles)
        log.info(f"Splitting into {nx}x{ny} = {nx * ny} tiles (overlap={overlap}m)")

        x_min, x_max = header.x_min, header.x_max
        y_min, y_max = header.y_min, header.y_max
        z_min, z_max = header.z_min, header.z_max

        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny

        tiles = []
        idx = 0

        for ix in range(nx):
            for iy in range(ny):
                # Core tile bounds (no overlap)
                tile_x_min = x_min + ix * dx
                tile_x_max = x_min + (ix + 1) * dx
                tile_y_min = y_min + iy * dy
                tile_y_max = y_min + (iy + 1) * dy

                core_bounds = Bounds(
                    mins=[tile_x_min, tile_y_min, z_min],
                    maxs=[tile_x_max, tile_y_max, z_max],
                )

                # Padded bounds (with overlap buffer)
                pad_x_min = max(tile_x_min - overlap, x_min)
                pad_x_max = min(tile_x_max + overlap, x_max)
                pad_y_min = max(tile_y_min - overlap, y_min)
                pad_y_max = min(tile_y_max + overlap, y_max)

                padded_bounds = Bounds(
                    mins=[pad_x_min, pad_y_min, z_min],
                    maxs=[pad_x_max, pad_y_max, z_max],
                )

                # Extract points using COPC spatial query (uses octree index)
                pts = reader.spatial_query(padded_bounds)
                n_points = len(pts.x)

                if n_points == 0:
                    log.warning(f"Tile {idx} is empty, skipping")
                    continue

                # Write tile to LAZ file (create fresh header — can't reuse COPC header)
                tile_path = os.path.join(work_dir, f"tile_{idx:04d}.laz")
                tile_header = laspy.LasHeader(
                    point_format=header.point_format,
                    version=header.version,
                )
                tile_header.offsets = header.offsets
                tile_header.scales = header.scales
                # Copy extra dimension definitions
                for dim in header.point_format.extra_dimensions:
                    tile_header.add_extra_dim(
                        laspy.ExtraBytesParams(name=dim.name, type=dim.dtype)
                    )
                tile_las = laspy.LasData(tile_header)
                tile_las.points = pts
                tile_las.write(tile_path)

                tile = Tile(
                    index=idx,
                    bounds=core_bounds,
                    file_path=tile_path,
                    point_count=n_points,
                    padded_bounds=padded_bounds,
                )
                tiles.append(tile)
                log.info(f"  Tile {idx}: {n_points:,} points -> {tile_path}")
                idx += 1

    log.info(f"Created {len(tiles)} tiles from {total_points:,} points")
    return tiles


def merge_tile_results(
    tiles: List[Tile],
    tile_results_dir: str,
    output_path: str,
    overlap: float = 2.0,
) -> str:
    """Merge inference results from multiple tiles back into a single file.

    For points in the overlap zone, prefer the prediction from the tile whose
    core (non-padded) bounds contain the point. This ensures each point gets
    exactly one prediction and avoids duplicate instance IDs at boundaries.

    Parameters
    ----------
    tiles : List[Tile]
        Tile metadata from split_copc_to_tiles().
    tile_results_dir : str
        Directory containing per-tile inference output files.
    output_path : str
        Path for the merged output file.
    overlap : float
        Same overlap value used during splitting.

    Returns
    -------
    str
        Path to the merged output file.
    """
    all_points = []
    all_sem = []
    all_inst = []
    instance_offset = 0  # Offset instance IDs to make them globally unique

    for tile in tiles:
        # Find the result file for this tile
        tile_name = Path(tile.file_path).stem
        result_candidates = [
            os.path.join(tile_results_dir, f"final_results/{tile_name}_out.laz"),
            os.path.join(tile_results_dir, f"final_results/{tile_name}_out.las"),
            os.path.join(tile_results_dir, f"final_results/{tile_name}.laz"),
        ]

        result_path = None
        for candidate in result_candidates:
            if os.path.exists(candidate):
                result_path = candidate
                break

        if result_path is None:
            log.warning(f"No result found for tile {tile.index}, skipping")
            continue

        # Read tile result
        las = laspy.read(result_path)
        x, y, z = las.x, las.y, las.z

        # Get predictions
        if "PredSemantic" in las.point_format.dimension_names:
            sem = np.array(las.PredSemantic)
        else:
            sem = np.zeros(len(x), dtype=np.int32)

        if "PredInstance" in las.point_format.dimension_names:
            inst = np.array(las.PredInstance)
        else:
            inst = np.zeros(len(x), dtype=np.int32)

        # Filter: only keep points within this tile's CORE bounds
        # (overlap zone points belong to the neighboring tile's core)
        core = tile.bounds
        in_core = (
            (x >= core.mins[0]) & (x < core.maxs[0]) &
            (y >= core.mins[1]) & (y < core.maxs[1])
        )

        x_core = x[in_core]
        y_core = y[in_core]
        z_core = z[in_core]
        sem_core = sem[in_core]
        inst_core = inst[in_core]

        # Offset instance IDs to make globally unique (skip 0 = unassigned)
        nonzero = inst_core > 0
        inst_core[nonzero] += instance_offset
        if nonzero.any():
            instance_offset = inst_core.max()

        all_points.append(np.column_stack([x_core, y_core, z_core]))
        all_sem.append(sem_core)
        all_inst.append(inst_core)

        log.info(f"  Tile {tile.index}: {in_core.sum():,} core points "
                 f"({len(x) - in_core.sum():,} overlap discarded)")

    # Concatenate all tiles
    points = np.vstack(all_points)
    sem = np.concatenate(all_sem)
    inst = np.concatenate(all_inst)

    log.info(f"Merged: {len(points):,} points, {instance_offset} unique instances")

    # Write output LAZ
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = points.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]

    # Add extra dimensions for predictions
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredSemantic", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.int32))

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.PredSemantic = sem
    las.PredInstance = inst
    las.write(output_path)

    log.info(f"Output: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split COPC file into tiles")
    parser.add_argument("input", help="Input .copc.laz file")
    parser.add_argument("-o", "--output", default="copc_tiles", help="Output directory")
    parser.add_argument("-n", "--num-tiles", type=int, default=4, help="Number of tiles")
    parser.add_argument("--overlap", type=float, default=2.0, help="Overlap buffer in meters")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    tiles = split_copc_to_tiles(args.input, args.output, args.num_tiles, args.overlap)
    for t in tiles:
        print(f"Tile {t.index}: {t.point_count:,} points -> {t.file_path}")

"""Point cloud density reduction (sparsification).

Reduces point cloud density to a target number of points per square meter
using random sampling, preserving the convex hull footprint.
"""

import argparse
import logging
import os
import random

import laspy
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm


class PointCloudSparsifier:
    """Reduce point density of a single LAS file to target pts/m^2."""

    def __init__(self, input_file, output_folder=None, target_density=10, verbose=False):
        self.input_file = input_file
        self.output_folder = output_folder or os.path.dirname(input_file)
        self.target_density = target_density
        self.output_file = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def _calculate_density(self, las):
        if hasattr(las, 'x') and las.x is not None:
            points_2d = np.vstack((las.x, las.y)).T
        else:
            points_2d = np.vstack((las.X, las.Y)).T
        hull = ConvexHull(points_2d)
        return len(points_2d) / hull.volume

    def process(self):
        os.makedirs(self.output_folder, exist_ok=True)
        in_file = laspy.read(self.input_file)
        density = self._calculate_density(in_file)
        self.logger.info(f"Current density: {density:.1f} pts/m^2")

        if density <= self.target_density:
            self.logger.info("Already sparser than target, skipping")
            filtered = in_file.points
        else:
            keep_count = int(len(in_file.x) * (self.target_density / density))
            indices = random.sample(range(len(in_file.x)), keep_count)
            filtered = in_file.points[indices]
            self.logger.info(f"Reduced {len(in_file.x)} -> {len(filtered)} points")

        out_file = laspy.create(point_format=in_file.point_format,
                                file_version=in_file.header.version)
        out_file.header = in_file.header
        out_file.points = filtered

        if self.output_file is None:
            basename = os.path.basename(self.input_file).replace(
                ".las", f"_sparse_{self.target_density}.las")
            self.output_file = os.path.join(self.output_folder, basename)
        out_file.write(self.output_file)


def sparsify_folder(input_folder, output_folder, target_density=10, verbose=False):
    """Sparsify all .las files in a folder tree."""
    paths = []
    for root, _, files in os.walk(input_folder):
        for f in files:
            if f.endswith(".las"):
                paths.append(os.path.join(root, f))

    for path in tqdm(paths, desc="Sparsifying"):
        rel = os.path.relpath(path, input_folder)
        out_dir = os.path.dirname(os.path.join(output_folder, rel))
        s = PointCloudSparsifier(path, out_dir, target_density, verbose)
        s.output_file = os.path.join(output_folder, rel)
        s.process()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Reduce point cloud density.')
    parser.add_argument("-i", "--input_folder", required=True)
    parser.add_argument("-o", "--output_folder", default=None)
    parser.add_argument("-d", "--target_density", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    sparsify_folder(args.input_folder, args.output_folder or args.input_folder,
                    args.target_density, args.verbose)

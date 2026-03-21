"""Convert LAS training data to PLY format with semantic label mapping.

Classification mapping (LAS -> semantic segmentation):
  0 -> 0 (unclassified)
  1 -> 1 (low vegetation -> non-tree)
  2 -> 1 (ground -> non-tree)
  3 -> 0 (outpoints -> unclassified)
  4 -> 2 (stem -> tree)
  5 -> 2 (live-branches -> tree)
  6 -> 2 (branches -> tree)
"""

import argparse
import csv
import random
from pathlib import Path

import laspy
import numpy as np
from plyfile import PlyElement, PlyData


def las_to_ply(las_file_path, ply_file_path, remove_ground=False,
               remove_lowveg=False, remove_outpoints=False):
    """Convert a LAS file to PLY with semantic label mapping.

    Args:
        las_file_path: Input LAS file.
        ply_file_path: Output PLY file path (base path; actual path may have suffix).
        remove_ground: Remove ground points (classification=2).
        remove_lowveg: Remove low vegetation points (classification=1).
        remove_outpoints: Remove outpoints (classification=3).

    Returns:
        Path where the PLY file was actually saved.
    """
    las = laspy.read(str(las_file_path))
    scale_x, scale_y, scale_z = las.header.scale
    X = las.X * scale_x
    Y = las.Y * scale_y
    Z = las.Z * scale_z

    n = las.X.size
    keep = np.full(n, True)

    suffix = ""
    if not (remove_ground or remove_lowveg or remove_outpoints):
        suffix = "_full"
    if remove_ground:
        keep &= (las.classification != 2)
        suffix += "_groundrm"
    if remove_lowveg:
        keep &= (las.classification != 1)
        suffix += "_lowvegrm"
    if remove_outpoints:
        keep &= (las.classification != 3)
        suffix += "_outpointsrm"

    cls = las.classification[keep]
    n_kept = cls.shape[0]

    data = np.zeros(n_kept, dtype=np.dtype([
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('intensity', 'f4'), ('semantic_seg', 'f4'), ('treeID', 'f4')
    ]))
    data['x'] = X.astype('f4')[keep]
    data['y'] = Y.astype('f4')[keep]
    data['z'] = Z.astype('f4')[keep]
    data['intensity'] = las.intensity.astype('f4')[keep]
    data['treeID'] = las.treeID.astype('f4')[keep]

    # Semantic label mapping
    sem = np.full(cls.shape, 20.0, dtype='f4')
    sem[cls == 0] = 0.0  # unclassified
    sem[cls == 1] = 1.0  # low veg -> non-tree
    sem[cls == 2] = 1.0  # ground -> non-tree
    sem[cls == 3] = 0.0  # outpoints -> unclassified
    sem[cls == 4] = 2.0  # stem -> tree
    sem[cls == 5] = 2.0  # live branches -> tree
    sem[cls == 6] = 2.0  # branches -> tree
    data['semantic_seg'] = sem

    ply_file_path = Path(ply_file_path)
    out_dir = ply_file_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ply_file_path.name

    el = PlyElement.describe(data, 'vertex')
    PlyData([el], byte_order='<').write(str(out_path))
    return out_path


def train_val_test_split(csv_path, val_fraction=0.25, seed=42):
    """Create train/val/test split from a train/test CSV metadata file.

    Args:
        csv_path: Path to data_split_metadata.csv with columns: path, region, split.
        val_fraction: Fraction of training data to use for validation.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (rel_path_list, forest_region_list, split_list).
    """
    random.seed(seed)

    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)  # skip header
        for row in reader:
            rows.append(row)

    train_count = sum(1 for r in rows if r[2] == "train")
    val_indices = set(random.sample(range(train_count), int(val_fraction * train_count)))

    rel_paths, regions, splits = [], [], []
    train_idx = 0
    for row in rows:
        rel_paths.append(row[0])
        regions.append(row[1])
        if row[2] == "test":
            splits.append("test")
        elif row[2] == "train":
            splits.append("val" if train_idx in val_indices else "train")
            train_idx += 1

    return rel_paths, regions, splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert LAS training data to PLY.')
    parser.add_argument('--las_dir', type=str, required=True,
                        help='Directory with LAS files and data_split_metadata.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for PLY files')
    parser.add_argument('--remove_ground', action='store_true')
    parser.add_argument('--remove_lowveg', action='store_true')
    parser.add_argument('--remove_outpoints', action='store_true')
    args = parser.parse_args()

    las_base = Path(args.las_dir)
    out_base = Path(args.output_dir)
    csv_path = las_base / 'data_split_metadata.csv'

    rel_paths, regions, splits = train_val_test_split(str(csv_path))
    test_paths = []

    for i, rel_path in enumerate(rel_paths):
        las_path = las_base / rel_path
        ply_name = f"{regions[i]}_{las_path.stem}_{splits[i]}.ply"
        ply_path = out_base / regions[i] / ply_name
        actual = las_to_ply(las_path, ply_path, args.remove_ground,
                            args.remove_lowveg, args.remove_outpoints)
        if splits[i] == "test":
            test_paths.append(str(actual))

    print("Test file paths (for eval.yaml fold):")
    for p in test_paths:
        print(f"  {p}")

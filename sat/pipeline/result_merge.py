"""Merge original point cloud with semantic and instance segmentation results.

After inference, the model outputs separate PLY files for instance and semantic
segmentation. This module merges them back with the original point cloud
(restoring UTM coordinates) and outputs a single LAS file per input.
"""

import argparse
import glob
import json
import os
import sys

import time

from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from sat.io.ply_io import ply_to_pandas, ply_to_numpy
from sat.io.las_io import pandas_to_las


class ResultMerger:
    """Merge a single triplet of point cloud + semantic + instance segmentation."""

    def __init__(self, point_cloud_path, semantic_path, instance_path,
                 output_path, verbose=False):
        self.point_cloud_path = point_cloud_path
        self.semantic_path = semantic_path
        self.instance_path = instance_path
        self.output_path = output_path
        self.verbose = verbose

    @staticmethod
    def _read_preds_numpy(file_path):
        """Read only the 'preds' column from a segmentation PLY using numpy."""
        structured, prop_names = ply_to_numpy(file_path)
        n_points = len(structured)
        if n_points == 0:
            return None, None, 0
        coords = np.column_stack([
            np.asarray(structured['x']),
            np.asarray(structured['y']),
            np.asarray(structured['z']),
        ]).astype(np.float32)
        preds = np.asarray(structured['preds']) if 'preds' in prop_names else None
        return coords, preds, n_points

    def merge(self):
        if self.verbose:
            print(f'Merging: {os.path.basename(self.point_cloud_path)}')

        t0 = time.time()

        # Read the original point cloud as a DataFrame (need all columns for LAS output)
        pc_df = ply_to_pandas(self.point_cloud_path)
        pc_df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)
        n_pc = len(pc_df)

        # Read segmentation results as numpy only (we just need preds)
        sem_coords, sem_preds, n_sem = self._read_preds_numpy(self.semantic_path)
        inst_coords, inst_preds, n_inst = self._read_preds_numpy(self.instance_path)

        t_read = time.time() - t0
        if self.verbose:
            print(f'  Read 3 files in {t_read:.1f}s (pc={n_pc:,}, sem={n_sem:,}, inst={n_inst:,})')

        t1 = time.time()

        # Build pc_coords lazily — only needed if KD-tree matching is required
        pc_coords = None
        need_kdtree_sem = sem_preds is not None and n_sem != n_pc
        need_kdtree_inst = inst_preds is not None and n_inst != n_pc
        if need_kdtree_sem or need_kdtree_inst:
            pc_coords = np.column_stack([
                pc_df['x'].values, pc_df['y'].values, pc_df['z'].values
            ]).astype(np.float32)

        # Attach semantic predictions
        if sem_preds is not None:
            if n_sem == n_pc:
                # Point counts match — assume preserved order, skip KD-tree
                pc_df['PredSemantic'] = sem_preds
            else:
                sem_tree = cKDTree(sem_coords)
                _, sem_idx = sem_tree.query(pc_coords, k=1, workers=-1)
                pc_df['PredSemantic'] = sem_preds[sem_idx]
        else:
            pc_df['PredSemantic'] = np.nan

        # Attach instance predictions
        if inst_preds is not None:
            if n_inst == n_pc:
                pc_df['PredInstance'] = inst_preds
            else:
                inst_tree = cKDTree(inst_coords)
                _, inst_idx = inst_tree.query(pc_coords, k=1, workers=-1)
                pc_df['PredInstance'] = inst_preds[inst_idx]
        else:
            pc_df['PredInstance'] = np.nan

        t_match = time.time() - t1
        if self.verbose:
            print(f'  Matched predictions in {t_match:.1f}s')

        merged_df = pc_df

        # Restore UTM coordinates
        min_values_path = self.point_cloud_path.replace('.ply', '_min_values.json')
        with open(min_values_path, 'r') as f:
            min_x, min_y, min_z = json.load(f)

        merged_df['x'] = merged_df['x'].astype(float) + min_x
        merged_df['y'] = merged_df['y'].astype(float) + min_y
        merged_df['z'] = merged_df['z'].astype(float) + min_z

        # Restore CRS if saved during coordinate transform
        crs_path = min_values_path.replace('_min_values.json', '_crs.wkt')
        crs_wkt = None
        if os.path.exists(crs_path):
            with open(crs_path, 'r') as f:
                crs_wkt = f.read().strip()
            if self.verbose:
                crs_short = crs_wkt[:60].replace('\n', ' ')
                print(f'  Restored CRS: {crs_short}...')

        # PredInstance: shift from 0-indexed to 1-indexed, fill NaN with 0
        if 'PredInstance' in merged_df.columns:
            merged_df['PredInstance'] = merged_df['PredInstance'] + 1
            merged_df['PredInstance'] = merged_df['PredInstance'].fillna(0)
            # Split oversized instances using CHM local maxima detection.
            # Note: we do NOT mask non-tree points from instances here.
            # The semantic head misclassifies ~57% of true tree points as
            # non-tree, so masking would destroy IoU with ground truth.
            # Instead, we let KNN propagate instances to all points and
            # rely on CHM splitting to separate merged trees.
            merged_df = self._split_large_instances(merged_df)

        if self.verbose:
            print(f'  Total merge: {time.time() - t0:.1f}s')

        return merged_df, crs_wkt

    @staticmethod
    def _split_large_instances(df, max_points=3000, chm_resolution=0.5,
                               local_max_window=3.0):
        """Split oversized instances using canopy-height local maxima detection.

        For dense forests where tree crowns interlock, DBSCAN fails because
        there's no gap between adjacent crowns. Instead, we:
        1. Build a canopy height model (CHM) grid from max Z values
        2. Find local maxima (tree tops) using a sliding window
        3. Assign each point to its nearest tree top in XY

        Args:
            max_points: Only split instances larger than this
            chm_resolution: Grid cell size for CHM (meters)
            local_max_window: Diameter of window for local maxima detection (meters)
        """
        from scipy.ndimage import maximum_filter

        if 'PredInstance' not in df.columns:
            return df

        inst_col = df['PredInstance'].values.copy()
        unique_ids = np.unique(inst_col)
        unique_ids = unique_ids[unique_ids > 0]
        next_id = int(inst_col.max()) + 1

        for inst_id in unique_ids:
            mask = inst_col == inst_id
            n_pts = mask.sum()
            if n_pts <= max_points:
                continue

            x = df.loc[mask, 'x'].values
            y = df.loc[mask, 'y'].values
            z = df.loc[mask, 'z'].values

            # Build CHM grid
            x_min, y_min = x.min(), y.min()
            nx = int(np.ceil((x.max() - x_min) / chm_resolution)) + 1
            ny = int(np.ceil((y.max() - y_min) / chm_resolution)) + 1

            chm = np.full((ny, nx), -np.inf)
            ix = ((x - x_min) / chm_resolution).astype(int).clip(0, nx - 1)
            iy = ((y - y_min) / chm_resolution).astype(int).clip(0, ny - 1)
            np.maximum.at(chm, (iy, ix), z)

            # Find local maxima (tree tops)
            win_size = max(3, int(np.ceil(local_max_window / chm_resolution)))
            if win_size % 2 == 0:
                win_size += 1
            local_max = maximum_filter(chm, size=win_size)
            peaks = (chm == local_max) & (chm > -np.inf)

            peak_iy, peak_ix = np.where(peaks)
            n_peaks = len(peak_iy)

            if n_peaks <= 1:
                continue  # Can't split

            # Convert peak grid coords back to world coords (center of cell)
            peak_x = peak_ix * chm_resolution + x_min + chm_resolution / 2
            peak_y = peak_iy * chm_resolution + y_min + chm_resolution / 2

            # Assign each point to nearest peak in XY
            peak_xy = np.column_stack([peak_x, peak_y])
            point_xy = np.column_stack([x, y])
            peak_tree = cKDTree(peak_xy)
            _, nearest_peak = peak_tree.query(point_xy, k=1)

            # Assign new instance IDs
            mask_indices = np.where(mask)[0]
            # Keep original ID for largest sub-cluster
            peak_ids, peak_counts = np.unique(nearest_peak, return_counts=True)
            largest_peak = peak_ids[np.argmax(peak_counts)]

            for peak_id in peak_ids:
                if peak_id == largest_peak:
                    continue
                sub_mask = nearest_peak == peak_id
                inst_col[mask_indices[sub_mask]] = next_id
                next_id += 1

        df['PredInstance'] = inst_col
        return df

    def save(self, merged_df, crs_wkt=None):
        for col in ('return_num', 'num_returns'):
            if col in merged_df:
                merged_df[col] = merged_df[col].clip(upper=7)

        pandas_to_las(merged_df, output_file_path=self.output_path,
                      do_compress=True, verbose=self.verbose, crs_wkt=crs_wkt)

    def run(self):
        merged_df, crs_wkt = self.merge()
        if self.output_path:
            self.save(merged_df, crs_wkt=crs_wkt)
        return merged_df


def _merge_triplet(pc, sem, inst, output_path, verbose):
    """Module-level function so joblib can serialize it for parallel execution."""
    ResultMerger(pc, sem, inst, output_path, verbose=verbose).run()


class FolderMerger:
    """Match and merge all point cloud + segmentation file triplets in folders."""

    def __init__(self, input_folder, segmented_folder, output_folder, verbose=False):
        self.input_folder = input_folder
        self.segmented_folder = segmented_folder
        self.output_folder = output_folder
        self.verbose = verbose
        os.makedirs(output_folder, exist_ok=True)

    def run(self):
        input_files = glob.glob(os.path.join(self.input_folder, '*.ply'))
        seg_files = glob.glob(os.path.join(self.segmented_folder, '*.ply'))

        input_map = {os.path.basename(f).split('.')[0]: f for f in input_files}
        sem_map = {os.path.basename(f).split('.')[0].replace('semantic_segmentation_', ''): f
                   for f in seg_files if 'semantic_segmentation_' in f}
        inst_map = {os.path.basename(f).split('.')[0].replace('instance_segmentation_', ''): f
                    for f in seg_files if 'instance_segmentation_' in f}

        matched = []
        for key in input_map:
            if key in sem_map and key in inst_map:
                matched.append((input_map[key], sem_map[key], inst_map[key]))

        if self.verbose:
            print(f'Found {len(matched)} matched triplets to merge')

        n_jobs = min(len(matched), os.cpu_count() or 4)
        if n_jobs > 1 and len(matched) > 1:
            Parallel(n_jobs=n_jobs)(
                delayed(_merge_triplet)(
                    pc, sem, inst,
                    os.path.join(self.output_folder, os.path.basename(pc).split('.')[0] + '.las'),
                    self.verbose,
                )
                for pc, sem, inst in matched
            )
        else:
            for pc, sem, inst in tqdm(matched, desc='Merging'):
                output_path = os.path.join(
                    self.output_folder,
                    os.path.basename(pc).split('.')[0] + '.las'
                )
                ResultMerger(pc, sem, inst, output_path, verbose=self.verbose).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge point cloud with segmentation results.')
    parser.add_argument('-i', '--input_data_folder_path', type=str, required=True)
    parser.add_argument('-s', '--segmented_data_folder_path', type=str, required=True)
    parser.add_argument('-o', '--output_data_folder_path', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    FolderMerger(
        args.input_data_folder_path,
        args.segmented_data_folder_path,
        args.output_data_folder_path,
        verbose=args.verbose,
    ).run()

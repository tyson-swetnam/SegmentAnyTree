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

        if self.verbose:
            print(f'  Total merge: {time.time() - t0:.1f}s')

        return merged_df, crs_wkt

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

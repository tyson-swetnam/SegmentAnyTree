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

import numpy as np
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm

from sat.io.ply_io import ply_to_pandas
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

    def _preprocess(self, file_path):
        df = ply_to_pandas(file_path)
        df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)
        df.sort_values(by=['x', 'y', 'z'], inplace=True)
        df['xyz_index'] = (df['x'].astype(str) + "_" +
                           df['y'].astype(str) + "_" +
                           df['z'].astype(str))
        df.set_index('xyz_index', inplace=True)
        return df

    def merge(self):
        if self.verbose:
            print(f'Merging: {os.path.basename(self.point_cloud_path)}')

        pc_df = self._preprocess(self.point_cloud_path)
        sem_df = self._preprocess(self.semantic_path)
        inst_df = self._preprocess(self.instance_path)

        sem_df.columns = [f'{col}_semantic_segmentation' for col in sem_df.columns]
        inst_df.columns = [f'{col}_instance_segmentation' for col in inst_df.columns]

        pc_dd = dd.from_pandas(pc_df, npartitions=48)
        sem_dd = dd.from_pandas(sem_df, npartitions=48)
        inst_dd = dd.from_pandas(inst_df, npartitions=48)

        merged_dd = pc_dd.join(sem_dd, how='outer').join(inst_dd, how='outer')
        merged_df = merged_dd.compute()

        # Drop duplicate coordinate columns from segmentation DataFrames
        for prefix in ('instance_segmentation', 'semantic_segmentation'):
            for coord in ('x', 'y', 'z'):
                col = f'{coord}_{prefix}'
                if col in merged_df.columns:
                    merged_df.drop(columns=[col], inplace=True)

        # Rename prediction columns
        merged_df.rename(columns={
            'preds_semantic_segmentation': 'PredSemantic',
            'preds_instance_segmentation': 'PredInstance',
        }, inplace=True)

        # Restore UTM coordinates
        min_values_path = self.point_cloud_path.replace('.ply', '_min_values.json')
        with open(min_values_path, 'r') as f:
            min_x, min_y, min_z = json.load(f)

        merged_df['x'] = merged_df['x'].astype(float) + min_x
        merged_df['y'] = merged_df['y'].astype(float) + min_y
        merged_df['z'] = merged_df['z'].astype(float) + min_z

        # PredInstance: shift from 0-indexed to 1-indexed, fill NaN with 0
        if 'PredInstance' in merged_df.columns:
            merged_df['PredInstance'] = merged_df['PredInstance'] + 1
            merged_df['PredInstance'] = merged_df['PredInstance'].fillna(0)

        return merged_df

    def save(self, merged_df):
        for col in ('return_num', 'num_returns'):
            if col in merged_df:
                merged_df[col] = merged_df[col].clip(upper=7)

        pandas_to_las(merged_df, output_file_path=self.output_path,
                      do_compress=True, verbose=self.verbose)

    def run(self):
        merged_df = self.merge()
        if self.output_path:
            self.save(merged_df)
        return merged_df


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

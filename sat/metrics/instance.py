"""Instance segmentation metrics: match predicted tree instances to ground truth via KNN.

Computes per-tree precision, recall, F1, IoU, and tree-level detection rates
(completeness, omission, commission).
"""

import argparse
import logging
import os

import laspy
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

logging.basicConfig(level=logging.INFO)


class InstanceSegmentationMetrics:
    """Compare predicted instance segmentation against ground truth labels."""

    GT_LABEL_NAME = 'treeID'
    TARGET_LABEL_NAME = 'PredInstance'

    def __init__(self, input_file_path, instance_segmented_file_path,
                 remove_ground=False, csv_file_name=None, verbose=False):
        self.input_file_path = input_file_path
        self.instance_segmented_file_path = instance_segmented_file_path
        self.remove_ground = remove_ground
        self.csv_file_name = csv_file_name
        self.verbose = verbose

        self.input_las = laspy.read(self.input_file_path)
        self.instance_segmented_las = laspy.read(self.instance_segmented_file_path)

        self.skip_flag = self._check_labels_exist()
        if not self.skip_flag:
            self.X_labels = self.input_las[self.GT_LABEL_NAME].astype(int)
            self.Y_labels = self.instance_segmented_las[self.TARGET_LABEL_NAME].astype(int) + 1
            self.Y_labels[self.Y_labels < 0] = 0
            self.dict_Y = self._do_knn_mapping()

    def _check_labels_exist(self):
        dims_in = self.input_las.header.point_format.dimension_names
        dims_seg = self.instance_segmented_las.header.point_format.dimension_names
        return (self.GT_LABEL_NAME not in dims_in or
                self.TARGET_LABEL_NAME not in dims_seg)

    def _do_knn_mapping(self):
        X = np.vstack((self.input_las.x, self.input_las.y, self.input_las.z)).T
        Y = np.vstack((self.instance_segmented_las.x,
                        self.instance_segmented_las.y,
                        self.instance_segmented_las.z)).T

        tree = KDTree(X, leaf_size=50)
        ind = tree.query(Y, k=1, return_distance=False)
        ind_labels_Y = self.X_labels[ind].reshape(-1)
        residual_ind = np.delete(np.arange(X.shape[0]), ind.reshape(-1))

        return {
            'X': X, 'Y': Y,
            'Y_labels': self.Y_labels,
            'ind_labels_Y': ind_labels_Y,
            'ind': ind,
            'residual_ind': residual_ind,
        }

    def compute_metrics(self):
        """Compute per-instance and aggregate metrics. Returns (per_instance, weighted, mean) dicts."""
        if self.skip_flag:
            return {}, {}, {}

        # Label matching logic (greedy, by GT class)
        dominant_sorted = self._get_dominant_labels_sorted()
        gt_classes = self._find_gt_classes()
        label_map = {}

        for gt_class in gt_classes:
            if not any(dominant_sorted.values()):
                break
            extracted = {k: {ik: iv for ik, iv in v.items() if ik == gt_class}
                         for k, v in dominant_sorted.items()}
            extracted = {k: v for k, v in extracted.items() if v}
            if not extracted:
                continue
            pred_key, gt_key = self._pick_dominant(extracted)
            label_map[pred_key] = gt_key
            dominant_sorted.pop(pred_key, None)
            for v in dominant_sorted.values():
                v.pop(gt_key, None)

        label_map = {int(k): int(v) for k, v in label_map.items()}

        # Per-instance metrics
        metric_dict = {}
        for pred_label, gt_label in label_map.items():
            mask_pred = self.Y_labels == pred_label
            matched_gt = self.dict_Y['ind_labels_Y'][mask_pred]
            tp = int(np.sum(matched_gt == gt_label))
            fp = int(np.sum(mask_pred)) - tp
            fn = int(np.sum(self.X_labels[self.dict_Y['residual_ind']] == gt_label))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

            metric_dict[str(pred_label)] = {
                'pred_label': pred_label, 'gt_label': gt_label,
                'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
                'precision': precision, 'recall': recall, 'f1_score': f1, 'IoU': iou,
            }

        # Aggregate
        params = ['precision', 'recall', 'f1_score', 'IoU']
        mean_dict = {p: 0 for p in params}
        if metric_dict:
            for v in metric_dict.values():
                for p in params:
                    mean_dict[p] += v[p]
            for p in params:
                mean_dict[p] /= len(metric_dict)

            # Tree-level detection
            gt_trees = set(np.unique(self.input_las[self.GT_LABEL_NAME]).astype(int)) - {0}
            correct = {v['gt_label'] for v in metric_dict.values() if v['IoU'] > 0.5}
            predicted = {v['gt_label'] for v in metric_dict.values()}
            mean_dict['detection_rate'] = len(correct) / len(gt_trees) if gt_trees else 0
            mean_dict['commission'] = len(predicted - correct) / len(predicted) if predicted else 0
            mean_dict['omission'] = len(gt_trees - correct) / len(gt_trees) if gt_trees else 0

        return metric_dict, {}, mean_dict

    def _get_dominant_labels_sorted(self):
        result = {}
        for label in np.unique(self.Y_labels):
            mask = self.Y_labels == label
            matched = self.dict_Y['ind_labels_Y'][mask]
            counts = {}
            for gl in np.unique(matched):
                counts[str(gl)] = int(np.sum(matched == gl))
            result[str(label)] = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        return dict(sorted(result.items(),
                           key=lambda x: max(x[1].values()) if x[1] else 0, reverse=True))

    def _find_gt_classes(self):
        labels, counts = np.unique(self.input_las[self.GT_LABEL_NAME].astype(int), return_counts=True)
        order = np.argsort(-counts)
        result = [str(l) for l in labels[order]]
        if self.remove_ground and '0' in result:
            result.remove('0')
        return result

    def _pick_dominant(self, extracted):
        best_key, best_gt, best_count = None, None, -1
        for k, v in extracted.items():
            for gt, count in v.items():
                if count > best_count:
                    best_key, best_gt, best_count = k, gt, count
        return best_key, best_gt

    def save_csv(self, metric_dict):
        if self.csv_file_name:
            pd.DataFrame(metric_dict).T.to_csv(self.csv_file_name, index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_path', required=True)
    parser.add_argument('--instance_segmented_file_path', required=True)
    parser.add_argument('--remove_ground', action='store_true')
    parser.add_argument('--csv_file_name', default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    m = InstanceSegmentationMetrics(
        args.input_file_path, args.instance_segmented_file_path,
        args.remove_ground, args.csv_file_name, args.verbose)
    per_inst, _, mean = m.compute_metrics()
    if args.verbose:
        for k, v in mean.items():
            print(f"{k}: {v:.4f}")
    if args.csv_file_name:
        m.save_csv(per_inst)

"""Drop-in compatibility shim for torch_points_kernels.

Provides ball_query, knn, and region_grow using torch_cluster and pure PyTorch,
eliminating the need for the unmaintained torch-points-kernels C++/CUDA package.

Usage: This module is installed as a fake 'torch_points_kernels' package via
the sys.modules trick in sat/compat/install.py, so existing code that does
`import torch_points_kernels as tp` works without modification.
"""

import torch
from torch_cluster import radius as _radius_search, knn as _knn_search
from sat.clustering.region_grow import region_grow


def instance_iou(predicted_clusters, instance_labels, batch):
    """Compute IoU between predicted clusters and ground truth instances.

    Parameters
    ----------
    predicted_clusters : List[Tensor]
        Each tensor contains point indices belonging to one predicted cluster.
    instance_labels : Tensor [N]
        Ground truth instance label per point.
    batch : Tensor [N]
        Batch index per point.

    Returns
    -------
    iou_matrix : Tensor [num_predicted, num_gt_instances]
        IoU between each predicted cluster and each GT instance.
    """
    # Find unique GT instances (excluding label 0 / -1 which are typically background)
    unique_instances = instance_labels.unique()
    unique_instances = unique_instances[unique_instances > 0]

    num_pred = len(predicted_clusters)
    num_gt = unique_instances.shape[0]

    if num_pred == 0 or num_gt == 0:
        return torch.zeros(max(num_pred, 1), max(num_gt, 1))

    iou_matrix = torch.zeros(num_pred, num_gt)

    # Pre-compute GT instance point sets
    gt_sets = []
    for gt_id in unique_instances:
        gt_sets.append(set((instance_labels == gt_id).nonzero(as_tuple=True)[0].tolist()))

    for i, cluster in enumerate(predicted_clusters):
        pred_set = set(cluster.tolist())
        for j, gt_set in enumerate(gt_sets):
            intersection = len(pred_set & gt_set)
            union = len(pred_set | gt_set)
            if union > 0:
                iou_matrix[i, j] = intersection / union

    return iou_matrix


def ball_query(radius, nsample, x, y, batch_x=None, batch_y=None):
    """Ball query: find up to nsample neighbors within radius.

    Parameters
    ----------
    radius : float
        Search radius.
    nsample : int
        Max number of neighbors.
    x : Tensor [N, 3]
        Database points.
    y : Tensor [M, 3]
        Query points.
    batch_x : Tensor [N], optional
    batch_y : Tensor [M], optional

    Returns
    -------
    idx : Tensor [M, nsample]
        Indices into x for each query point in y. Padded with -1.
    dist : Tensor [M, nsample]
        Squared distances. Padded with 0.
    """
    N = x.shape[0]
    M = y.shape[0]

    if batch_x is None:
        batch_x = torch.zeros(N, dtype=torch.long, device=x.device)
    if batch_y is None:
        batch_y = torch.zeros(M, dtype=torch.long, device=y.device)

    # torch_cluster.radius returns (row, col) where row indexes y and col indexes x
    row, col = _radius_search(x, y, r=radius, batch_x=batch_x, batch_y=batch_y,
                              max_num_neighbors=nsample)

    # Build padded output [M, nsample]
    idx = torch.full((M, nsample), -1, dtype=torch.long, device=x.device)
    dist = torch.zeros(M, nsample, dtype=torch.float, device=x.device)

    # Count neighbors per query point
    counts = torch.zeros(M, dtype=torch.long, device=x.device)

    for i in range(row.shape[0]):
        r = row[i].item()
        c = col[i].item()
        cnt = counts[r].item()
        if cnt < nsample:
            idx[r, cnt] = c
            dist[r, cnt] = ((y[r] - x[c]) ** 2).sum()
            counts[r] += 1

    return idx, dist


def knn(x, y, k, batch_x=None, batch_y=None):
    """K-nearest neighbors search.

    Parameters
    ----------
    x : Tensor [N, 3]
        Database points.
    y : Tensor [M, 3]
        Query points.
    k : int
        Number of neighbors.
    batch_x : Tensor [N], optional
    batch_y : Tensor [M], optional

    Returns
    -------
    idx : Tensor [M, k]
        Indices into x for each query point.
    dist : Tensor [M, k]
        Squared distances.
    """
    N = x.shape[0]
    M = y.shape[0]

    if batch_x is None:
        batch_x = torch.zeros(N, dtype=torch.long, device=x.device)
    if batch_y is None:
        batch_y = torch.zeros(M, dtype=torch.long, device=y.device)

    # torch_cluster.knn returns (row, col) pairs
    assign = _knn_search(x, y, k, batch_x=batch_x, batch_y=batch_y)
    row, col = assign[0], assign[1]

    idx = col.reshape(M, k)
    # Compute squared distances
    dist = ((y.unsqueeze(1).expand(-1, k, -1) - x[idx]) ** 2).sum(-1)

    return idx, dist


# Expose everything that existing code expects
__all__ = ["ball_query", "knn", "region_grow", "instance_iou"]

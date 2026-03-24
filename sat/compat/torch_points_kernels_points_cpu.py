"""Compatibility shim for torch_points_kernels.points_cpu submodule.

Provides ball_query and dense_knn using torch_cluster.
"""

import torch
from torch_cluster import radius as _radius_search, knn as _knn_search


def ball_query(radius, nsample, x, y, batch_x=None, batch_y=None):
    """Ball query on CPU: find up to nsample neighbors within radius.

    Parameters
    ----------
    radius : float
    nsample : int
    x : Tensor [N, 3] — database points
    y : Tensor [M, 3] — query points
    batch_x, batch_y : optional batch tensors

    Returns
    -------
    idx : Tensor [M, nsample] (padded with -1)
    dist : Tensor [M, nsample] (squared distances, padded with 0)
    """
    N = x.shape[0]
    M = y.shape[0]

    if batch_x is None:
        batch_x = torch.zeros(N, dtype=torch.long, device=x.device)
    if batch_y is None:
        batch_y = torch.zeros(M, dtype=torch.long, device=y.device)

    row, col = _radius_search(x, y, r=radius, batch_x=batch_x, batch_y=batch_y,
                              max_num_neighbors=nsample)

    idx = torch.full((M, nsample), -1, dtype=torch.long, device=x.device)
    dist = torch.zeros(M, nsample, dtype=torch.float, device=x.device)
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


def dense_knn(x, y, k, batch_x=None, batch_y=None):
    """Dense k-nearest neighbors search.

    Parameters
    ----------
    x : Tensor [N, 3]
    y : Tensor [M, 3]
    k : int

    Returns
    -------
    idx : Tensor [M, k]
    dist : Tensor [M, k]
    """
    N = x.shape[0]
    M = y.shape[0]

    if batch_x is None:
        batch_x = torch.zeros(N, dtype=torch.long, device=x.device)
    if batch_y is None:
        batch_y = torch.zeros(M, dtype=torch.long, device=y.device)

    assign = _knn_search(x, y, k, batch_x=batch_x, batch_y=batch_y)
    row, col = assign[0], assign[1]

    idx = col.reshape(M, k)
    dist = ((y.unsqueeze(1).expand(-1, k, -1) - x[idx]) ** 2).sum(-1)

    return idx, dist

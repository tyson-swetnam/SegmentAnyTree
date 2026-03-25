"""Pure PyTorch reimplementation of region_grow from torch-points-kernels.

Replaces the unmaintained C++/CUDA dependency with torch_cluster.radius()
for ball queries and union-find for connected components.

This runs on CPU, matching the original usage pattern where all call sites
pass .cpu() tensors.
"""

from typing import List, Optional

import numpy as np
import numba
import torch
from torch_cluster import radius


@numba.jit(nopython=True)
def _union_find_batch(parent, rank, edge_target, edge_source):
    """Process all edges through union-find in compiled code."""
    for i in range(edge_target.shape[0]):
        a, b = edge_target[i], edge_source[i]
        # inline find with path compression for a
        ra = a
        while parent[ra] != ra:
            parent[ra] = parent[parent[ra]]
            ra = parent[ra]
        # inline find with path compression for b
        rb = b
        while parent[rb] != rb:
            parent[rb] = parent[parent[rb]]
            rb = parent[rb]
        # union by rank
        if ra != rb:
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1


@numba.jit(nopython=True)
def _flatten_roots(parent):
    """Flatten all parent pointers to roots in compiled code."""
    n = len(parent)
    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = i
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        roots[i] = r
    return roots


def region_grow(
    positions: torch.Tensor,
    predicted_labels: torch.Tensor,
    batch: torch.Tensor,
    ignore_labels: Optional[torch.Tensor] = None,
    radius_value: float = 0.03,
    nsample: int = 200,
    min_cluster_size: int = 10,
    # Accept 'radius' as kwarg for backward compat with existing call sites
    **kwargs,
) -> List[torch.Tensor]:
    """Region growing clustering on point clouds.

    Groups points into clusters by:
    1. Ball query to find neighboring points within a radius
    2. Only connecting points with matching semantic labels
    3. Ignoring points with labels in ignore_labels
    4. Connected components via union-find

    Parameters
    ----------
    positions : Tensor [N, 3]
        Point positions (CPU).
    predicted_labels : Tensor [N]
        Predicted semantic label per point (CPU).
    batch : Tensor [N]
        Batch index per point (CPU).
    ignore_labels : Tensor, optional
        Labels to exclude from clustering (e.g., stuff/background classes).
    radius_value : float
        Ball query search radius.
    nsample : int
        Maximum neighbors per point (limits computation on dense regions).
    min_cluster_size : int
        Minimum number of points to form a valid cluster.

    Returns
    -------
    List[Tensor]
        List of 1D tensors, each containing point indices for one cluster.
    """
    # Handle the 'radius' keyword used by existing call sites
    if "radius" in kwargs:
        radius_value = kwargs["radius"]

    N = positions.shape[0]
    if N == 0:
        return []

    # Ensure CPU and correct dtypes
    positions = positions.float().cpu()
    predicted_labels = predicted_labels.long().cpu()
    batch = batch.long().cpu()
    if batch.dim() > 1:
        batch = batch.squeeze(-1)

    # Build mask for valid (non-ignored) points
    if ignore_labels is not None:
        ignore_labels = ignore_labels.long().cpu()
        valid_mask = ~torch.isin(predicted_labels, ignore_labels)
    else:
        valid_mask = torch.ones(N, dtype=torch.bool)

    valid_indices = torch.where(valid_mask)[0]
    if valid_indices.numel() == 0:
        return []

    # Ball query using torch_cluster.radius()
    # Returns (row, col) pairs where row indexes into y (query) and col into x (database)
    # We query valid points against valid points
    valid_pos = positions[valid_indices]
    valid_batch = batch[valid_indices]
    valid_labels = predicted_labels[valid_indices]

    # radius() returns edges as (target_idx, source_idx) in the valid_pos space
    edge_target, edge_source = radius(
        valid_pos, valid_pos, r=radius_value,
        batch_x=valid_batch, batch_y=valid_batch,
        max_num_neighbors=nsample,
    )

    # Filter: only keep edges where both endpoints have the same label
    same_label = valid_labels[edge_target] == valid_labels[edge_source]
    # Remove self-loops
    not_self = edge_target != edge_source
    keep = same_label & not_self
    edge_target = edge_target[keep]
    edge_source = edge_source[keep]

    # Union-find connected components (Numba-accelerated)
    n_valid = valid_indices.shape[0]
    parent = np.arange(n_valid, dtype=np.int64)
    rank = np.zeros(n_valid, dtype=np.int64)

    # Convert to numpy for Numba
    et = edge_target.numpy().astype(np.int64)
    es = edge_source.numpy().astype(np.int64)

    _union_find_batch(parent, rank, et, es)
    roots = torch.from_numpy(_flatten_roots(parent))

    # Group points by root, mapping back to original indices
    clusters = []
    unique_roots = torch.unique(roots)
    for root in unique_roots:
        members = valid_indices[roots == root]
        if members.shape[0] >= min_cluster_size:
            clusters.append(members)

    return clusters

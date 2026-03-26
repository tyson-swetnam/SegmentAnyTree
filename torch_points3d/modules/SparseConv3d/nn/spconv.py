"""SpConv v2.x backend wrapper for the SparseConv3d abstraction layer.

Provides the same API as torchsparse.py and minkowski.py:
    Conv3d, Conv3dTranspose, BatchNorm, ReLU, cat, SparseTensor
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv


class _SparseTensorWrapper:
    """Thin wrapper around spconv.SparseConvTensor that adds .F and .C properties
    expected by the SparseConv3d application layer."""

    def __init__(self, sct):
        self._sct = sct

    @property
    def F(self):
        return self._sct.features

    @property
    def C(self):
        # Coordinates are stored in [batch, x, y, z] order (matching MinkowskiEngine).
        # No reordering needed since we pass [x, y, z] directly in SparseTensor().
        return self._sct.indices  # [N, 4] int32: [batch, x, y, z]

    def __add__(self, other):
        """Element-wise feature addition for matching sparsity patterns."""
        other_sct = other._sct if isinstance(other, _SparseTensorWrapper) else other
        new_feats = self._sct.features + other_sct.features
        return _SparseTensorWrapper(self._sct.replace_feature(new_feats))

    def __radd__(self, other):
        return self.__add__(other)

    def __getattr__(self, name):
        return getattr(self._sct, name)


# Global counter for auto-generating unique indice_keys to pair encoder/decoder layers
_indice_key_counter = 0


def _next_indice_key():
    global _indice_key_counter
    _indice_key_counter += 1
    return f"sp_conv_{_indice_key_counter}"


def reset_indice_keys():
    """Reset the indice_key counter. Call before each forward pass."""
    global _indice_key_counter
    _indice_key_counter = 0


class Conv3d(nn.Module):
    """Sparse 3D convolution. Uses SubMConv3d for stride=1, SparseConv3d for stride>1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride == 1:
            self.conv = spconv.SubMConv3d(
                in_channels, out_channels,
                kernel_size=kernel_size, dilation=dilation, bias=bias,
            )
        else:
            # Add padding to match MinkowskiEngine behavior where strided conv
            # doesn't reduce spatial dimensions beyond the stride factor.
            # Without padding, kernel_size=3 + stride=2 collapses small dimensions to 0.
            padding = kernel_size // 2
            self._indice_key = _next_indice_key()
            self.conv = spconv.SparseConv3d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias,
                padding=padding, indice_key=self._indice_key,
            )

    @property
    def kernel(self):
        """Alias for .weight to match MinkowskiEngine convention used by weight_initialization()."""
        return self.conv.weight

    def forward(self, x):
        if isinstance(x, _SparseTensorWrapper):
            return _SparseTensorWrapper(self.conv(x._sct))
        return _SparseTensorWrapper(self.conv(x))


class Conv3dTranspose(nn.Module):
    """Sparse 3D transposed convolution.

    Uses SparseInverseConv3d for stride>1 to invert a paired encoder SparseConv3d.
    SpConv requires SparseInverseConv3d (not SparseConvTranspose3d) when reusing
    indice_keys from strided encoder convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride == 1:
            self.conv = spconv.SubMConv3d(
                in_channels, out_channels,
                kernel_size=kernel_size, dilation=dilation, bias=bias,
            )
        else:
            padding = kernel_size // 2
            self.conv = spconv.SparseConvTranspose3d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias,
                padding=padding,
            )

    @property
    def kernel(self):
        """Alias for .weight to match MinkowskiEngine convention used by weight_initialization()."""
        return self.conv.weight

    def forward(self, x):
        if isinstance(x, _SparseTensorWrapper):
            return _SparseTensorWrapper(self.conv(x._sct))
        return _SparseTensorWrapper(self.conv(x))


class BatchNorm(nn.Module):
    """Batch normalization for sparse tensors. Operates on the feature dimension."""

    def __init__(self, num_features: int, *, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        if isinstance(x, _SparseTensorWrapper):
            new_feats = self.bn(x._sct.features)
            return _SparseTensorWrapper(x._sct.replace_feature(new_feats))
        # Assume raw SparseConvTensor
        new_feats = self.bn(x.features)
        return _SparseTensorWrapper(x.replace_feature(new_feats))

    def __repr__(self):
        return self.bn.__repr__()


class ReLU(nn.Module):
    """ReLU activation for sparse tensors."""

    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        if isinstance(x, _SparseTensorWrapper):
            new_feats = self.relu(x._sct.features)
            return _SparseTensorWrapper(x._sct.replace_feature(new_feats))
        new_feats = self.relu(x.features)
        return _SparseTensorWrapper(x.replace_feature(new_feats))


def pair_encoder_decoder(down_modules, up_modules):
    """Pair SparseConv3d (encoder) and SparseInverseConv3d (decoder) layers via indice_key.

    In a UNet, encoder down_modules[0..N] pair with decoder up_modules[N..0] in reverse.
    Each strided Conv3d in the encoder generates an indice_key when created.
    This function copies those keys to the corresponding SparseInverseConv3d layers.
    """
    # Collect strided Conv3d layers from encoder (in order)
    enc_keys = []
    for mod in down_modules.children():
        for m in mod.modules():
            if isinstance(m, Conv3d) and m.stride > 1 and hasattr(m, '_indice_key'):
                enc_keys.append(m._indice_key)

    # Collect SparseInverseConv3d layers from decoder (in order = reverse of encoder)
    dec_convs = []
    for mod in up_modules.children():
        for m in mod.modules():
            if isinstance(m, Conv3dTranspose) and m.stride > 1:
                dec_convs.append(m)

    # With SparseConvTranspose3d (no indice_key pairing), skip connection filtering
    # handles coordinate alignment between encoder and decoder.
    # No rebuild needed.
    pass

    # Also pair ResNetUp modules that use lazy SpConv building
    from torch_points3d.modules.SparseConv3d.modules import ResNetUp
    dec_ups = []
    for mod in up_modules.children():
        if isinstance(mod, ResNetUp) and getattr(mod, '_spconv_mode', False):
            dec_ups.append(mod)

    # Also collect kernel sizes from encoder strided convs
    enc_kernel_sizes = []
    for mod in down_modules.children():
        for m in mod.modules():
            if isinstance(m, Conv3d) and m.stride > 1 and hasattr(m, '_indice_key'):
                ks = m.conv.kernel_size
                enc_kernel_sizes.append(ks[0] if hasattr(ks, '__len__') else ks)

    for i, up_mod in enumerate(dec_ups):
        if i < len(enc_keys):
            up_mod._paired_indice_key = enc_keys[-(i + 1)]
        if i < len(enc_kernel_sizes):
            up_mod._paired_kernel_size = enc_kernel_sizes[-(i + 1)]


def _filter_to_coords(output, reference):
    """Filter a SparseTensorWrapper to only keep coordinates present in reference.

    This is used after SparseConvTranspose3d to clip the output back to the
    encoder's coordinate set, preventing exponential voxel growth in the UNet decoder.
    """
    out_sct = output._sct if isinstance(output, _SparseTensorWrapper) else output
    ref_sct = reference._sct if isinstance(reference, _SparseTensorWrapper) else reference

    out_indices = out_sct.indices  # [M, 4]
    ref_indices = ref_sct.indices  # [N, 4]

    # Hash both coordinate sets for fast lookup
    def _hash(idx):
        return idx[:, 0].long() * (2**48) + idx[:, 1].long() * (2**32) + idx[:, 2].long() * (2**16) + idx[:, 3].long()

    out_hash = _hash(out_indices)
    ref_hash_set = set(_hash(ref_indices).tolist())

    # Find which output indices exist in the reference
    mask = torch.tensor([h.item() in ref_hash_set for h in out_hash],
                        dtype=torch.bool, device=out_sct.features.device)

    filtered_feats = out_sct.features[mask]
    filtered_indices = out_indices[mask]

    new_sct = spconv.SparseConvTensor(
        filtered_feats, filtered_indices,
        out_sct.spatial_shape, out_sct.batch_size,
    )
    # Propagate indice_dict
    for k, v in out_sct.indice_dict.items():
        new_sct.indice_dict[k] = v
    for k, v in ref_sct.indice_dict.items():
        if k not in new_sct.indice_dict:
            new_sct.indice_dict[k] = v

    return _SparseTensorWrapper(new_sct)


def cat(*args):
    """Concatenate sparse tensors along the feature dimension.

    In a UNet decoder, the upsampled tensor and skip connection should share the same
    sparsity pattern (same active voxels), so we concatenate features directly.
    If sparsity patterns differ, we fall back to coordinate-aligned concatenation.
    """
    # Flatten: cat(a, b) or cat((a, b))
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = args[0]

    tensors = [a._sct if isinstance(a, _SparseTensorWrapper) else a for a in args]

    # Fast path: same number of active sites — assume matching sparsity
    if all(t.features.shape[0] == tensors[0].features.shape[0] for t in tensors):
        cat_features = torch.cat([t.features for t in tensors], dim=1)
        result = tensors[0].replace_feature(cat_features)
        # Propagate indice_dict from all input tensors
        for t in tensors[1:]:
            if hasattr(t, 'indice_dict'):
                for k, v in t.indice_dict.items():
                    if k not in result.indice_dict:
                        result.indice_dict[k] = v
        return _SparseTensorWrapper(result)

    # Slow path: different sparsity patterns — align by coordinates
    # Build a coordinate-to-index mapping from each tensor
    ref = tensors[0]
    all_indices = torch.cat([t.indices for t in tensors], dim=0)
    unique_indices, inverse = torch.unique(all_indices, dim=0, return_inverse=True)
    total_channels = sum(t.features.shape[1] for t in tensors)
    merged_features = torch.zeros(
        unique_indices.shape[0], total_channels,
        dtype=ref.features.dtype, device=ref.features.device,
    )
    offset = 0
    idx_offset = 0
    for t in tensors:
        n = t.features.shape[0]
        c = t.features.shape[1]
        merged_features[inverse[idx_offset:idx_offset + n], offset:offset + c] = t.features
        offset += c
        idx_offset += n

    new_sct = spconv.SparseConvTensor(
        merged_features, unique_indices,
        ref.spatial_shape, ref.batch_size,
    )
    # Propagate indice_dict from input tensors so SparseInverseConv3d can find
    # the paired encoder's indice data downstream
    for t in tensors:
        if hasattr(t, 'indice_dict'):
            for k, v in t.indice_dict.items():
                if k not in new_sct.indice_dict:
                    new_sct.indice_dict[k] = v
    return _SparseTensorWrapper(new_sct)


def SparseTensor(feats, coordinates, batch, device=torch.device("cpu")):
    """Create a SpConv SparseConvTensor from features, coordinates, and batch indices.

    Parameters
    ----------
    feats : Tensor [N, C]
    coordinates : Tensor [N, 3] (x, y, z)
    batch : Tensor [N] or [N, 1]
    device : torch.device
    """
    if batch.dim() == 1:
        batch = batch.unsqueeze(-1)
    # Keep coordinates in [x, y, z] order to match MinkowskiEngine's convention.
    # SpConv's internal dimension labels (z, y, x) are arbitrary — what matters is
    # that the kernel weights and coordinates use the same spatial ordering as training.
    # ME trains with [x, y, z], so we pass [x, y, z] directly.
    coords_xyz = coordinates.int()

    # Shift all coordinates away from boundaries by a margin.
    # SpConv's strided conv validation rejects points that would fall outside
    # the output spatial volume. A margin ensures no point is near any edge.
    margin = 32
    coords_xyz = coords_xyz + margin

    indices = torch.cat([batch.int(), coords_xyz], dim=-1).contiguous()

    # Spatial shape: max coord + 1 (margin already added above), rounded to next power of 2 (min 512).
    import math
    max_coords = (coords_xyz.max(0).values + 1).tolist()
    spatial_shape = [max(2 ** math.ceil(math.log2(max(s, 1))), 512) for s in max_coords]

    batch_size = int(batch.max().item()) + 1

    sct = spconv.SparseConvTensor(
        feats.to(device), indices.to(device),
        spatial_shape, batch_size,
    )
    return _SparseTensorWrapper(sct)

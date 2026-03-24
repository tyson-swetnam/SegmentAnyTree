# SegmentAnyTree (SAT) - Tree instance segmentation from 3D point cloud data

# Install compatibility shims for removed C++/CUDA dependencies
# (torch_points_kernels -> pure PyTorch replacements)
from sat.compat.install import install_shims as _install_shims
_install_shims()

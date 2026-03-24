"""Install compatibility shims into sys.modules.

Call sat.compat.install.install_shims() early in the application to make
'import torch_points_kernels' resolve to our pure-PyTorch replacement.
"""

import sys


def install_shims():
    """Register the torch_points_kernels compatibility shim and submodules."""
    if "torch_points_kernels" not in sys.modules:
        try:
            # Try the real package first
            import torch_points_kernels  # noqa: F401
        except ImportError:
            # Install our shim as the top-level package
            from sat.compat import torch_points_kernels as _shim
            sys.modules["torch_points_kernels"] = _shim

            # Install submodule shims (torch_points_kernels.points_cpu)
            from sat.compat import torch_points_kernels_points_cpu as _cpu_shim
            sys.modules["torch_points_kernels.points_cpu"] = _cpu_shim
            _shim.points_cpu = _cpu_shim

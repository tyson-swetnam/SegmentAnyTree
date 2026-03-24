import os
import sys
import logging
import importlib

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
sys.path.insert(0, ROOT)

log = logging.getLogger(__name__)

# Import a backend for documentation and linting purposes
try:
    from .spconv import *  # type: ignore
except Exception:
    try:
        from .torchsparse import *  # type: ignore
    except Exception:
        try:
            from .minkowski import *  # type: ignore
        except Exception:
            pass


__all__ = ["cat", "Conv3d", "Conv3dTranspose", "ReLU", "SparseTensor", "BatchNorm"]

def backend_valid(_backend):
    return _backend in {"spconv", "torchsparse", "minkowski"}

# Detect which backend was successfully imported at the top of this file
sp3d_backend = None
_imported_ok = "Conv3d" in dir() and locals().get("Conv3d") is not None
if _imported_ok:
    try:
        import spconv.pytorch
        sp3d_backend = "spconv"
    except ImportError:
        try:
            import torchsparse
            sp3d_backend = "torchsparse"
        except ImportError:
            try:
                import MinkowskiEngine
                sp3d_backend = "minkowski"
            except ImportError:
                pass

if sp3d_backend is None:
    # No backend available — set symbols to None so downstream code gets clear errors
    cat = None
    Conv3d = None
    Conv3dTranspose = None
    ReLU = None
    SparseTensor = None
    BatchNorm = None

def get_backend():
    return sp3d_backend

def set_backend(_backend):
    """ Use this method to switch sparse backend dynamically. When importing this module with a wildcard such as
    from torch_points3d.modules.SparseConv3d.nn import *
    make sure that you import it again after calling this method.


    Parameters
    ----------
    backend : str
        "torchsparse" or "minkowski"
    """
    assert backend_valid(_backend)
    try:
        modules = importlib.import_module("." + _backend, __name__)  # noqa: F841
        global sp3d_backend
        sp3d_backend = _backend
    except:
        log.exception("Could not import %s backend for sparse convolutions" % _backend)
    for val in __all__:
        exec("globals()['%s'] = modules.%s" % (val, val))

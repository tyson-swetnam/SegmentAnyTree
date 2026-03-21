"""
Central path resolution for SegmentAnyTree.

All paths are derived from environment variables with sensible defaults.
Inside Docker, SAT_ROOT=/opt/segmentanytree. Outside Docker, it defaults
to the repository root (detected via git or file structure).
"""

import os
from pathlib import Path


def _find_repo_root() -> Path:
    """Walk up from this file to find the repository root (contains torch_points3d/)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "torch_points3d").is_dir():
            return current
        current = current.parent
    return Path.cwd()


def get_sat_root() -> Path:
    """Project root directory. Contains model_file/, conf/, torch_points3d/, etc."""
    return Path(os.environ.get("SAT_ROOT", str(_find_repo_root())))


def get_sat_data() -> Path:
    """Data directory for input/output. In Docker this is /data (a mount point)."""
    default = str(get_sat_root() / "data")
    return Path(os.environ.get("SAT_DATA", default))


def get_sat_model() -> Path:
    """Model checkpoint directory. Contains PointGroup-PAPER.pt."""
    default = str(get_sat_root() / "model_file")
    return Path(os.environ.get("SAT_MODEL", default))


def get_sat_cache() -> Path:
    """Temporary files directory. Cleared between runs."""
    return Path(os.environ.get("SAT_CACHE", "/tmp/sat_cache"))


def get_input_dir() -> Path:
    """Default input directory within SAT_DATA."""
    return get_sat_data() / "input"


def get_output_dir() -> Path:
    """Default output directory within SAT_DATA."""
    return get_sat_data() / "output"


def ensure_dirs(*dirs: Path) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

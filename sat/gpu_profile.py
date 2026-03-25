"""Auto-detect GPU memory and select optimized configuration profiles.

Profiles scale cluster_nsample, num_workers, and batch_size based on
available VRAM. Higher VRAM allows more neighbors in region_grow clustering
and larger batch sizes for training.
"""

import os
import torch
import logging

log = logging.getLogger(__name__)

PROFILES = {
    80: {
        "name": "a100_80gb",
        "cluster_nsample": 200,
        "cluster_type": 1,
        "num_workers": 8,
        "batch_size_train": 12,
        "batch_size_eval": 1,
    },
    40: {
        "name": "a100_40gb",
        "cluster_nsample": 128,
        "cluster_type": 1,
        "num_workers": 6,
        "batch_size_train": 6,
        "batch_size_eval": 1,
    },
    24: {
        "name": "rtx3090_24gb",
        "cluster_nsample": 96,
        "cluster_type": 1,
        "num_workers": 4,
        "batch_size_train": 4,
        "batch_size_eval": 1,
    },
    0: {
        "name": "default",
        "cluster_nsample": 32,
        "cluster_type": 1,
        "num_workers": 2,
        "batch_size_train": 2,
        "batch_size_eval": 1,
    },
}


def _get_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return props.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def select_profile(gpu_memory_gb: float = None) -> dict:
    if gpu_memory_gb is None:
        gpu_memory_gb = _get_gpu_memory_gb()

    for min_gb in sorted(PROFILES.keys(), reverse=True):
        if gpu_memory_gb >= min_gb:
            profile = PROFILES[min_gb]
            log.info(f"GPU profile: {profile['name']} ({gpu_memory_gb:.0f} GB detected)")
            profile = dict(profile)
            # Scale num_workers by available CPUs (don't exceed CPU count / 2)
            max_workers = max((os.cpu_count() or 4) // 2, 2)
            profile["num_workers"] = min(profile["num_workers"], max_workers)
            return profile

    return dict(PROFILES[0])


def apply_profile_to_config(cfg, profile: dict = None):
    from omegaconf import open_dict

    if profile is None:
        profile = select_profile()

    with open_dict(cfg):
        if hasattr(cfg, "models"):
            model_cfg = getattr(cfg.models, "PointGroup-PAPER", None)
            if model_cfg is not None and model_cfg.get("cluster_nsample", 32) == 32:
                model_cfg.cluster_nsample = profile["cluster_nsample"]
                log.info(f"  cluster_nsample -> {profile['cluster_nsample']}")

        cfg.num_workers = profile["num_workers"]
        log.info(f"  num_workers -> {profile['num_workers']}")

        # Apply batch_size if present in profile and config
        if hasattr(cfg, "batch_size"):
            if hasattr(cfg, "training") or "train" in str(cfg.get("job_name", "")):
                cfg.batch_size = profile.get("batch_size_train", cfg.batch_size)
            else:
                cfg.batch_size = profile.get("batch_size_eval", cfg.batch_size)
            log.info(f"  batch_size -> {cfg.batch_size}")

    return cfg

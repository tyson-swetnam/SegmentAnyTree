import sat.compat.install  # noqa: E402,F401 — must run before torch_points3d imports
sat.compat.install.install_shims()

import os
import torch
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer
import logging


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    # Auto-tune config for available GPU
    from sat.gpu_profile import select_profile, apply_profile_to_config
    apply_profile_to_config(cfg, select_profile())

    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    # Initialize DDP if launched with torchrun
    if "LOCAL_RANK" in os.environ:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")

    trainer = Trainer(cfg)
    trainer._setup_ddp()
    trainer.train()

    # Cleanup DDP
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()

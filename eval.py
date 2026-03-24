import sat.compat.install  # noqa: E402,F401 — must run before torch_points3d imports
sat.compat.install.install_shims()

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer


@hydra.main(config_path="conf", config_name="eval")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))

    # Auto-tune config for available GPU
    from sat.gpu_profile import select_profile, apply_profile_to_config
    apply_profile_to_config(cfg, select_profile())

    trainer = Trainer(cfg)
    trainer.eval(stage_name = "test")
    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()

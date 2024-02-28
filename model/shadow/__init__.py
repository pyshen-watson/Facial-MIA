import torch.nn as nn
from configs import Config
from typing import Callable
from .shadowModule import ShadowModule


def get_shadow(cfg: Config, backbone:nn.Module, callback: Callable) -> ShadowModule:
    
    if cfg.shadow_ckpt_path is not None:
        shadow = ShadowModule.load_from_checkpoint(cfg.shadow_ckpt_path, cfg=cfg, backbone=backbone, callback=callback)
        print(f'Loaded the checkpoint from {cfg.shadow_ckpt_path}')
        return shadow
    return ShadowModule(cfg, backbone, callback)
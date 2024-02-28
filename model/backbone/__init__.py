import torch
import torch.nn as nn
from configs import Config
from .mobilefacenet import get_mbf, get_mbf_large
from .mobilefacenet_dp import get_dpmbf, get_dpmbf_large

backbone_getter = {
    'mbf': get_mbf,
    'mbf_large': get_mbf_large,
    'dp_mbf': get_dpmbf,
    'dp_mbf_large': get_dpmbf_large
}


def get_backbone(cfg: Config) -> nn.Module:
    
    getter = backbone_getter.get(cfg.model_name)
    
    if getter is None:
        raise ValueError(f'Unknown backbone name: {cfg.model_name}. The valid names are {list(backbone_getter.keys())}.')

    backbone = getter(cfg.fp16, cfg.output_size)

    if cfg.target_ckpt_path is not None:
        try:
            backbone.load_state_dict(torch.load(cfg.target_ckpt_path))
            print(f'Loaded the checkpoint from {cfg.target_ckpt_path}')
        except:
            raise ValueError(f'Failed to load the checkpoint from {cfg.target_ckpt_path}. Please check if the file exists or matches the model {cfg.model_name}')

    return backbone
import torch
from configs import Config
from .mobilefacenet import get_mbf, get_mbf_large, MobileFaceNet

def get_backbone(cfg: Config) -> MobileFaceNet:
    """
    This function returns a backbone model based on the given model_name. It has parameters model_name, fp16, num_features, and pretrain. The default values for fp16 and pretrain are False and True, respectively. The default value for num_features is 512. It returns the backbone model.
    """
    
    if cfg.model_name == 'mbf':
        backbone = get_mbf(cfg.fp16, cfg.output_size)

    elif cfg.model_name == 'mbf_large':
        backbone = get_mbf_large(cfg.fp16, cfg.output_size)
    
    else:
        raise ValueError(f'Unknown model name: {cfg.model_name}')

    if cfg.bkb_ckpt_path is not None:
        backbone.load_state_dict(torch.load(cfg.bkb_ckpt_path))

    return backbone
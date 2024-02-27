from configs import Config
from .mobilefacenet import MobileFaceNet
from .backbone import get_backbone as __get_backbone__
from .reconstructor import ReconstructModule

def get_backbone(cfg: Config) -> MobileFaceNet:
    """
    This function returns a backbone model based on the given model_name. It has parameters model_name, fp16, num_features, and pretrain. The default values for fp16 and pretrain are False and True, respectively. The default value for num_features is 512. It returns the backbone model.
    """
    return __get_backbone__(cfg)
    

def get_reconstructor(cfg: Config) -> ReconstructModule:
    """
    This function returns a reconstructor model based on the given config. It has parameters cfg and returns the reconstructor model.
    """
    if cfg.rec_ckpt_path is not None:
        print(f'Loading reconstructor from {cfg.rec_ckpt_path}')
        return ReconstructModule.load_from_checkpoint(cfg.rec_ckpt_path, cfg=cfg)
    return ReconstructModule(cfg)
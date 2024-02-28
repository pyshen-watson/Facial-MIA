from torch.utils.data import  Dataset
from configs.config import Config
from .cffp import CFFP_DataModule
from .common import *


def get_raw_dataset(cfg: Config) -> Dataset:

    if cfg.shadow_dataset == 'lfw':
        from .lfw_raw import LFWRawDataset
        return LFWRawDataset(cfg.input_size)
    
    elif cfg.shadow_dataset == 'ms1mv3':
        pass

    else:
        raise ValueError(f'Dataset {cfg.shadow_dataset} not found')

def get_dataModule(cfg: Config) -> CFFP_DataModule:
    
    if cfg.shadow_dataset not in ['lfw', 'ms1mv3']:
        raise ValueError(f'Dataset {cfg.shadow_dataset} not found')
    return CFFP_DataModule(cfg)
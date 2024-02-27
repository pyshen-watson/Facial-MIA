from torch.utils.data import  Dataset
from configs.config import Config
from .cffp import CFFP_DataModule
from .common import *


def get_raw_dataset(cfg: Config) -> Dataset:

    if cfg.dataset == 'lfw':
        from .lfw_raw import LFWRawDataset
        return LFWRawDataset(cfg.input_size)
    
    elif cfg.dataset == 'ms1mv3':
        pass

    else:
        raise ValueError(f'Dataset {cfg.dataset} not found')

def get_dataModule(cfg: Config) -> CFFP_DataModule:
    
    if cfg.dataset not in ['lfw', 'ms1mv3']:
        raise ValueError(f'Dataset {cfg.dataset} not found')
    return CFFP_DataModule(cfg)
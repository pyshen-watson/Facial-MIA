from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Experiment
    exp_name: str = 'LWF'
    
    # Data
    dataset: str = 'lfw'
    cache_dir: str = '.cache'
    split_ratio: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    input_size: int = 112
    
    # Model
    model_name: str = 'mbf_large'
    fp16: bool = False
    output_size: int = 512
    bkb_ckpt_path: str = None
    rec_ckpt_path: str = None
    
    # Training
    batch_size: int = 128
    num_workers: int = 16
    lr: float = 1e-3
    max_epochs: int = 500
    dssim_weight: float = 0.1
    id_weight: float = 0.005
    
    @property
    def cache_path(self):
        return f'{self.cache_dir}/{self.dataset}.npy'



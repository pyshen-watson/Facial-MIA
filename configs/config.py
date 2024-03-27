from typing import List
from dataclasses import dataclass, field

@dataclass
class Config:
    # Experiment
    exp_name: str = 'Title'
    
    # Data
    target_dataset: str = 'lfw'
    shadow_dataset: str = 'lfw'
    input_size: int = 112
    split_ratio: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    
    # Model
    model_name: str = 'mbf_large'
    fp16: bool = False
    output_size: int = 512
    target_ckpt_path: str = ""
    shadow_ckpt_path: str = ""
    
    # Training
    batch_size: int = 64
    num_workers: int = 16
    lr: float = 1e-3
    max_epochs: int = 250
    dssim_weight: float = 0.1
    id_weight: float = 0.1 # 0.005 in oringinal paper


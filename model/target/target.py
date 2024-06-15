import torch
import torch.nn as nn

from enum import Enum
from dataclasses import dataclass, field
from colorama import Fore, Style

from .mbf_v1 import get_mbf_large as mbf_large_v1
from .mbf_v2 import get_dpmbf_large as mbf_large_v2
from .mbf_v3 import get_noisy_mbf_large as mbf_large_v3


class TargetType(Enum):
    MBF_LARGE_V1 = "mbf_large_v1"
    MBF_LARGE_V2 = "mbf_large_v2"
    MBF_LARGE_V3 = "mbf_large_v3"

    @staticmethod
    def keys():
        '''Just return the names of the target models for error message.'''
        return [member.name for member in TargetType]


@dataclass(eq=False)
class TargetModel(nn.Module):
    '''
    This class is a base wrapper for target model to support some common methods.
    It should be initialized with one of the `TargetType`.
    '''

    model_type: TargetType
    fp16: bool = False
    num_features: int = 512
    backbone: nn.Module = field(init=False)

    def __post_init__(self):
        """Initialize the target model with the given model type."""

        super(TargetModel, self).__init__()

        if self.model_type == TargetType.MBF_LARGE_V1:
            self.backbone = mbf_large_v1(self.fp16, self.num_features)

        elif self.model_type == TargetType.MBF_LARGE_V2:
            self.backbone = mbf_large_v2(self.fp16, self.num_features)

        elif self.model_type == TargetType.MBF_LARGE_V3:
            self.backbone = mbf_large_v3(self.fp16, self.num_features)
            
        else:
            msg = f"✗ Unknown model name: {self.model_type}. The target model should be one of {TargetType.keys()}."
            raise ValueError(Fore.RED + msg + Style.RESET_ALL)

    def load(self, ckpt_path: str):
        """
        This method loads the checkpoint from the given path and return the model itself.
        `ckpt_path` is the path to the checkpoint file in .pt or .pth format.
        """

        try:
            self.backbone.load_state_dict(torch.load(ckpt_path))
            print(Fore.GREEN + f"✓ Loaded the checkpoint from {ckpt_path}" + Style.RESET_ALL)
            return self

        except Exception as _:
            msg = f"✗ Failed to load the checkpoint from {ckpt_path}. Please check if the file exists or matches the model {self.model_type}"
            raise ValueError(Fore.RED + msg + Style.RESET_ALL)

    def freeze(self):
        ''' This method freezes the backbone of the target model. '''
        for param in self.backbone.parameters():
            param.requires_grad = False
        return self

    def forward(self, x):
        return self.backbone(x)

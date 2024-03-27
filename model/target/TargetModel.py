import torch
import torch.nn as nn

from enum import Enum
from dataclasses import dataclass, field

from .mbf_v1 import get_mbf_large as mbf_large_v1
from .mbf_v2 import get_dpmbf_large as mbf_large_v2


class TargetType(Enum):
    MBF_LARGE_V1 = "mbf_large_v1"
    MBF_LARGE_V2 = "mbf_large_v2"

    @staticmethod
    def keys():
        return [member.name for member in TargetType]


@dataclass(eq=False)
class TargetModel(nn.Module):

    fp16: bool = False
    model_type: TargetType = TargetType.MBF_LARGE_V1
    num_features: int = 512
    backbone: nn.Module = field(init=False)

    def __post_init__(self):
        """Initialize the target model with the given model type."""

        super(TargetModel, self).__init__()

        if self.model_type == TargetType.MBF_LARGE_V1:
            self.backbone = mbf_large_v1(self.fp16, self.num_features)

        elif self.model_type == TargetType.MBF_LARGE_V2:
            self.backbone = mbf_large_v2(self.fp16, self.num_features)

        else:
            raise ValueError(
                f"Unknown model name: {self.model_type}. The target model should be one of {TargetType.keys()}."
            )

    def load(self, ckpt_path: str):

        try:
            self.backbone.load_state_dict(torch.load(ckpt_path))
            print(f"Loaded the checkpoint from {ckpt_path}")
            return self

        except Exception as _:
            raise ValueError(
                f"Failed to load the checkpoint from {ckpt_path}. Please check if the file exists or matches the model {self.model_type}"
            )

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        return self

    def forward(self, x):
        return self.backbone(x)

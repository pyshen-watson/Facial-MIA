import torch
import torch.nn as nn

from enum import Enum
from dataclasses import dataclass, field

from .idiap import IdiapModel, IdiapLoss


class ShadowType(Enum):
    IDIAP = "idiap"

    @staticmethod
    def keys():
        return [member.name for member in ShadowType]


@dataclass(eq=False)
class ShadowModel(nn.Module):

    model_type: ShadowType = ShadowType.IDIAP
    num_features: int = 512
    dssim_weight: float = 0.1
    id_weight: float = 0.1

    backbone: nn.Module = field(init=False)
    loss_fn: nn.Module = field(init=False)

    def __post_init__(self):
        """Initialize the shadow model with the given model type."""

        super(ShadowModel, self).__init__()

        if self.model_type == ShadowType.IDIAP:
            self.backbone = IdiapModel(self.num_features)
            self.loss_fn = IdiapLoss(self.dssim_weight, self.id_weight)

        else:
            raise ValueError( f"Unknown model name: {self.model_type}. The target model should be one of {ShadowType.keys()}." )

    def load(self, ckpt_path: str):

        try:
            self.backbone.load_state_dict(torch.load(ckpt_path))
            print(f"Loaded the checkpoint from {ckpt_path}")
            return self

        except Exception as _:
            raise ValueError( f"Failed to load the checkpoint from {ckpt_path}. Please check if the file exists or matches the model {self.model_type}" )

    def set_target(self, target: nn.Module):
        self.loss_fn.target = target
        return self

    def forward(self, x):
        return self.backbone(x)

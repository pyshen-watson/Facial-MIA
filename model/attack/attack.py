import torch
import torch.nn as nn
from enum import Enum
from colorama import Fore, Style
from dataclasses import dataclass, field
from .idiap import IdiapModel, IdiapLoss


class AttackType(Enum):
    IDIAP = "idiap"

    @staticmethod
    def keys():
        '''This method returns the names of the shadow models for error message.'''
        return [member.name for member in AttackType]


@dataclass(eq=False)
class AttackModel(nn.Module):

    model_type: AttackType
    num_features: int = 512
    dssim_weight: float = 0.1
    id_weight: float = 0.1

    backbone: nn.Module = field(init=False)
    loss_fn: nn.Module = field(init=False)

    def __post_init__(self):
        """Initialize the shadow model with the given model type."""

        super(AttackModel, self).__init__()

        if self.model_type == AttackType.IDIAP:
            self.backbone = IdiapModel(self.num_features)
            self.loss_fn = IdiapLoss(self.dssim_weight, self.id_weight)

        else:
            msg = f"✗ Unknown model name: {self.model_type}. The target model should be one of {AttackType.keys()}."
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

    def set_target(self, target: nn.Module):
        """Set the target model for calculating the loss."""
        self.loss_fn.target = target
        return self

    def forward(self, x):
        return self.backbone(x)

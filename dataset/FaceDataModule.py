from typing import Tuple
from dataclasses import dataclass, field

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split


@dataclass
class FaceDataModule(pl.LightningDataModule):

    dataset: Dataset
    split_ratio: Tuple[float, float, float] = field(default_factory=lambda:[0.8, 0.1, 0.1])
    batch_size: int = 64
    num_workers: int = 16
    
    train_set: Dataset = field(init=False)
    val_set: Dataset = field(init=False)
    test_set: Dataset = field(init=False)

    def __post_init__(self):
        super(FaceDataModule, self).__init__()
        self.train_set, self.val_set, self.test_set = random_split(self.dataset, self.split_ratio)

    def X_dataloader(self, dataset: Dataset, shuffle=False):
        return DataLoader(
            dataset,
            self.batch_size,
            shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self.X_dataloader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self.X_dataloader(self.val_set)

    def test_dataloader(self):
        return self.X_dataloader(self.test_set)

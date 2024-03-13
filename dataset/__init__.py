from typing import List
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
    
from face_data import FaceDataset, get_dataset
from configs.config import Config


def get_dataModule(cfg: Config):
    
    dataset = get_dataset(cfg.shadow_dataset, cfg.input_size)
    split_ratio = cfg.split_ratio
    
    return FaceDataModule(dataset, split_ratio)

class FaceDataModule(pl.LightningDataModule):

    def __init__(self, dataset: FaceDataset, split_ratio: List[float]):
        super(FaceDataModule, self).__init__()
        self.train_set, self.val_set, self.test_set = random_split(dataset, split_ratio)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
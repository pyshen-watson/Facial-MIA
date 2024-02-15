import pytorch_lightning as pl
from torch.utils.data import DataLoader

class FRDataModule(pl.LightningDataModule):
    
    def __init__(self, data, batch_size=32, num_workers=4):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Prepare the dataset and annotations
        self.train_set = self.data[0]
        self.val_set = self.data[1]
        self.test_set = self.data[2]

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

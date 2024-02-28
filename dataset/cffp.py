import numpy as np
import pytorch_lightning as pl
from configs.config import Config

from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

####################################################################################################
# The CFFP dataset is a dataset that contains the Crop Face and Feature Pair
# It is used for training the reconstruct model
####################################################################################################    


class CFFP_DataModule(pl.LightningDataModule):

    def __init__(self, cfg: Config):
        super(CFFP_DataModule, self).__init__()
        dataset = CFFP_Dataset(cfg)
        self.cfg = cfg
        self.train_set, self.val_set, self.test_set = random_split(
            dataset, cfg.split_ratio
        )

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


class CFFP_Dataset(Dataset):

    def __init__(self, cfg: Config):

        self.data = np.load(cfg.cache_path, allow_pickle=True).item()
        self.transform = T.Compose([T.ToPILImage(), T.Resize(cfg.input_size), T.ToTensor()])

    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, idx):
        img = self.data["img"][idx]
        feat = self.data["feat"][idx]
        img = self.transform(img)
        return img, feat


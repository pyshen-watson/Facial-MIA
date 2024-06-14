import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from functools import partial


def split_data(ds: Dataset, train_ratio=0.8):
    """
    Split the dataset `ds` into train, validation, and test sets.
    `train_ratio` should be between 0~1.
    """
    val_ratio = (1 - train_ratio) / 2
    train_len = int(len(ds) * train_ratio)
    val_len = int(len(ds) * val_ratio)
    test_len = len(ds) - train_len - val_len
    return random_split(ds, [train_len, val_len, test_len])

def get_transform(size=112, is_train=False):
    if is_train:
        return T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(size, scale=(0.8, 1.0)),
            T.ToTensor(),
        ])
    else:
        return T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
        ])
        
class FaceDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str, size=112, batch_size=64, num_workers=12, train_ratio=0.8):
        super(FaceDataModule, self).__init__()

        # Load and split the dataset
        ds = ImageFolder(root_dir)
        self.train_ds, self.val_ds, self.test_ds = split_data(ds, train_ratio)

        # Set the transforms        
        self.train_ds.dataset.transform = get_transform(size, is_train=True)
        self.val_ds.dataset.transform = get_transform(size, is_train=False)
        self.test_ds.dataset.transform = get_transform(size, is_train=False)

        # This is a partial function that will be used to create dataloaders
        self.create_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=num_workers)
        
    def train_dataloader(self):
        return self.create_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.val_ds)

    def test_dataloader(self):
        return self.create_dataloader(self.test_ds)

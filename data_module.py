import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.dataset import get_lfw_people

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

# Below is an example of how to use the data module

if __name__ == '__main__':
    
    import torchvision.transforms as T
    
    data = get_lfw_people(train_ratio=0.8, 
                          transform=T.Compose([
                              T.Resize(112), 
                              T.ToTensor()]))

    lfw_data_module = FRDataModule(
        data=data, 
        batch_size=32, 
        num_workers=4)
    
    train_loader = lfw_data_module.train_dataloader()
    x, y = next(iter(train_loader))
    print(x.dtype, y.dtype)
    print(x.shape, y.shape)
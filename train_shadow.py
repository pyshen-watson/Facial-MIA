import pytorch_lightning as pl
from model import TargetModel, TargetType
from model import ShadowModel, ShadowType
from model import ReconstructModule
from dataset import HDF5Dataset, HDF5DatasetType
from dataset import FaceDataModule

pl.seed_everything(42, workers=True)

target = TargetModel(model_type=TargetType.MBF_LARGE_V1).load('weights/backbone/mbf_large_v1.pt').freeze().eval()
shadow = ShadowModel(model_type=ShadowType.IDIAP).set_target(target)
model = ReconstructModule(target, shadow)

dataset = HDF5Dataset(HDF5DatasetType.CELEBA)
dataModule = FaceDataModule(dataset, batch_size=160)
train_loader = dataModule.train_dataloader()
val_loader = dataModule.val_dataloader()

trainer = pl.Trainer(max_epochs=250, benchmark=True)
trainer.fit(model, train_loader, val_loader)

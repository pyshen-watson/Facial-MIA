import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import TargetModel, TargetType
from model import ShadowModel, ShadowType
from model import ReconstructModule
from dataset import HDF5Dataset, HDF5DatasetType
from dataset import FaceDataModule

pl.seed_everything(42, workers=True)

target = TargetModel(model_type=TargetType.MBF_LARGE_V1).load('weights/backbone/mbf_large_v1.pt').freeze().eval()
shadow = ShadowModel(model_type=ShadowType.IDIAP).set_target(target)
model = ReconstructModule(target, shadow)

dataset = HDF5Dataset(HDF5DatasetType.LFW)
train_ratio = 0.8
val_ratio = (1-train_ratio)/2
test_ratio = val_ratio
dataModule = FaceDataModule(dataset, batch_size=128, split_ratio=[train_ratio, val_ratio, test_ratio])
train_loader = dataModule.train_dataloader()
val_loader = dataModule.val_dataloader()

checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, strict=False, verbose=True, mode='min')

trainer = pl.Trainer(max_epochs=200, benchmark=True, callbacks=[checkpoint_callback, early_stopping_callback])
trainer.fit(model, train_loader, val_loader)

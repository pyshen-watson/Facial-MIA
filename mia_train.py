import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import TargetModel, TargetType
from model import ShadowModel, ShadowType
from model import ReconstructModule
from dataset import FaceDataModule

pl.seed_everything(42, workers=True)

target = TargetModel(model_type=TargetType.MBF_LARGE_V1).load('weights/backbone/mbf_large_v1.pt').freeze().eval()
shadow = ShadowModel(model_type=ShadowType.IDIAP).set_target(target)
model = ReconstructModule(target, shadow)

dm = FaceDataModule('../MIA/cropped_data/lfw', batch_size=128, train_ratio=0.8)
checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, strict=False, verbose=True, mode='min')

trainer = pl.Trainer(max_epochs=200, benchmark=True, callbacks=[checkpoint_callback, early_stopping_callback])
trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import TargetModel, TargetType
from model import AttackModel, AttackType
from model import ModelInversionAttackModule
from dataset import FaceDataModule

pl.seed_everything(42, workers=True)

target = TargetModel(model_type=TargetType.MBF_LARGE_V1).load('weights/target/mbf_large_v1.pt').freeze().eval()
shadow = AttackModel(model_type=AttackType.IDIAP, dssim_weight=0.25, id_weight=0.25).set_target(target)
model = ModelInversionAttackModule(target, shadow)

dm = FaceDataModule('../MIA/cropped_data/lfw', batch_size=160, train_ratio=0.8)
checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, strict=False, verbose=True, mode='min')

trainer = pl.Trainer(max_epochs=200, benchmark=True, callbacks=[checkpoint_callback, early_stopping_callback])
trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

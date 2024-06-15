import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import TargetModel, TargetType
from model import AttackModel, AttackType
from model import ModelInversionAttackModule
from dataset import FaceDataModule

pl.seed_everything(42, workers=True)

target = TargetModel(model_type=TargetType.MBF_LARGE_V1).load('weights/target/mbf_large_v1.pt')
attack = AttackModel(model_type=AttackType.IDIAP, id_weight=0.2)
mia_model = ModelInversionAttackModule(target, attack)

dm = FaceDataModule('../cropped_data/lfw', batch_size=50, train_ratio=0.8)
checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, strict=False, verbose=True, mode='min')

trainer = pl.Trainer(devices=[1], max_epochs=1, benchmark=True, callbacks=[checkpoint_callback, early_stopping_callback])
trainer.fit(mia_model, dm.train_dataloader(), dm.val_dataloader())

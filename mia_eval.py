import pytorch_lightning as pl

from model import TargetModel, TargetType
from model import AttackModel, AttackType
from model import ModelInversionAttackModule
from dataset import FaceDataModule

pl.seed_everything(42, workers=True)

target = TargetModel(model_type=TargetType.MBF_LARGE_V1).load('weights/target/mbf_large_v1.pt')
attack = AttackModel(model_type=AttackType.IDIAP).load('mbf_large_v1+idiap.pt')
mia_model = ModelInversionAttackModule(target, attack)

dm = FaceDataModule('../cropped_data/lfw', batch_size=128, train_ratio=0.8)

trainer = pl.Trainer(max_epochs=200, benchmark=True)
trainer.test(mia_model, dm.test_dataloader())

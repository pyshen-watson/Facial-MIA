import pytorch_lightning as pl

from model import TargetModel, TargetType
from model import ShadowModel, ShadowType
from model import ReconstructModule
from dataset import FaceDataModule

pl.seed_everything(42, workers=True)

target = TargetModel(model_type=TargetType.MBF_LARGE_V1).load('weights/backbone/mbf_large_v1.pt').freeze().eval()
shadow = ShadowModel(model_type=ShadowType.IDIAP).load('weights/shadow/mbf_large_v1+idiap.pt').set_target(target)
model = ReconstructModule(target, shadow)

dm = FaceDataModule('../MIA/cropped_data/lfw', batch_size=128, train_ratio=0.8)

trainer = pl.Trainer(max_epochs=200, benchmark=True)
trainer.test(model, dm.test_dataloader())

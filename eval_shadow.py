import pytorch_lightning as pl
from model import TargetModel, TargetType
from model import ShadowModel, ShadowType
from model import ReconstructModule
from dataset import HDF5Dataset, HDF5DatasetType
from dataset import FaceDataModule

pl.seed_everything(42, workers=True)

target = TargetModel(model_type=TargetType.MBF_LARGE_V1).load('weights/backbone/mbf_large_v1.pt').freeze().eval()
shadow = ShadowModel(model_type=ShadowType.IDIAP).load("weights/shadow/mbf_large_v1+idiap(celeba).pt").set_target(target)
model = ReconstructModule(target, shadow)

dataset = HDF5Dataset(HDF5DatasetType.LFW)
dataModule = FaceDataModule(dataset, batch_size=160)
test_loader = dataModule.test_dataloader()

trainer = pl.Trainer(max_epochs=1, benchmark=True)
trainer.test(model, test_loader)

import torch
import torch.nn as nn
import pytorch_lightning as pl
from data_module import FRDataModule
from torchvision import transforms as T
from dataset.dataset import get_lfw_people
from models.model import get_backbone, FaceRecognitionModule
from configs.base import config

def get_data_module(config) -> FRDataModule:
    
    transform = T.Compose([
                            T.Resize(config.input_size), 
                            T.ToTensor()
                        ])
    
    data = get_lfw_people(config.train_ratio, transform)

    lfw_data_module = FRDataModule(data, 
        config.batch_size, 
        config.num_workers) 
    
    return lfw_data_module

def get_model(config) -> nn.Module:

    backbone = get_backbone(
        config.model_name, 
        config.fp16, 
        config.num_features, 
        config.pretrain,
    )

    model = FaceRecognitionModule(backbone, config)

    return model

def summary_model(model):
    summary = pl.utilities.model_summary.ModelSummary
    print(summary(model))

def main():
    data_module = get_data_module(config)
    model = get_model(config)
    trainer = pl.Trainer(max_epochs=config.max_epochs)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

if __name__ == '__main__':
    main()

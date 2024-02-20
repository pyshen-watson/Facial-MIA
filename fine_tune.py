import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser

from dataset.dataset import get_lfw_people
from dataset.data_module import FRDataModule
from models.model import get_backbone, FaceRecognitionModule

def get_config():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='path to config file')
    args = parser.parse_args()
    config_path = args.config
    config_name = 'configs.' + config_path.split('/')[-1].split('.')[0]
    
    try:
        config_module = __import__(config_name, fromlist=['config'])
        return config_module.config
    except:
        raise ValueError(f'Config file {config_name} not found')    
        
def get_data_module(config) -> FRDataModule:
    
    data = get_lfw_people(config)

    lfw_data_module = FRDataModule(
        data, 
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
    
    config = get_config()
    data_module = get_data_module(config)
    model = get_model(config)
    logger = TensorBoardLogger(save_dir='lightning_logs', name=config.exp_name)
    trainer = Trainer(max_epochs=config.max_epochs, logger=logger)
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

if __name__ == '__main__':
    main()

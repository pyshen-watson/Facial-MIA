import pytorch_lightning as pl
from argparse import ArgumentParser

from configs import get_config
from dataset import get_dataModule, draw_rows
from model import get_backbone, get_shadow


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', default='default', type=str, help='[default/lfw/test]')
    args = parser.parse_args()
    return args


def main():

    arg = get_args()
    config = get_config(arg.config)
    
    ############################ Load Resources ############################
    
    datamodule = get_dataModule(config)
    train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()
    test_loader  = datamodule.test_dataloader()

    backbone = get_backbone(config)
    shadow = get_shadow(config, backbone, callback=draw_rows)

    ############################ Train and Test ############################

    logger = pl.loggers.TensorBoardLogger( 'lightning_logs', name=config.exp_name)
    trainer = pl.Trainer(max_epochs=config.max_epochs, logger=logger)
    trainer.fit(shadow, train_loader, val_loader)
    trainer.predict(shadow, test_loader)

if __name__ == '__main__':
    main()

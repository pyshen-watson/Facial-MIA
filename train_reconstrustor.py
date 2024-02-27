import pytorch_lightning as pl
from argparse import ArgumentParser

from configs import get_config
from dataset import get_dataModule
from model import get_reconstructor


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', default='default', type=str, help='[default/lfw/ms1mv3]')
    args = parser.parse_args()
    return args


def main():

    ############################ Load Resources ############################
    arg = get_args()
    config = get_config(arg.config)
    datamodule = get_dataModule(config)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    reconstructor = get_reconstructor(config)

    ############################ Train and Test ############################


    logger = pl.loggers.TensorBoardLogger( 'lightning_logs', name=config.exp_name)
    trainer = pl.Trainer(max_epochs=config.max_epochs, logger=logger)
    trainer.fit(reconstructor, train_loader, val_loader)


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import pytorch_lightning as pl

from pathlib import Path
from pytorch_msssim import SSIM
from dataclasses import asdict

from .backbone import get_backbone
from .utils import draw_rows
from .decoder import Decoder
from configs import Config

class DSSIMLoss(nn.Module):
    
    def __init__(self):
        super(DSSIMLoss, self).__init__()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, x1, x2):
        dssim = (1-self.ssim(x1, x2)) / 2
        return dssim
    
class IDLoss(nn.Module):
        
    def __init__(self, feat_extractor):
        super(IDLoss, self).__init__()
        self.feat_extractor = feat_extractor
        self.mse = nn.MSELoss()

    def forward(self, rec_img, ori_feat):
        rec_feat = self.feat_extractor(rec_img)
        return self.mse(rec_feat, ori_feat)

class ReconstructModule(pl.LightningModule):

    def __init__(self, cfg: Config):

        super().__init__()
        self.save_hyperparameters(asdict(cfg))
        self.cfg = cfg

        self.encoder = get_backbone(cfg)
        self.decoder = Decoder(cfg.output_size)
        
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.mse = nn.MSELoss()
        self.dssim = DSSIMLoss()
        self.idloss = IDLoss(self.encoder)
        
        self.lr = cfg.lr
        self.alpha = cfg.dssim_weight
        self.beta = cfg.id_weight

    def forward(self, feat):
        feat = feat.unsqueeze(-1).unsqueeze(-1)
        img_rec = self.decoder(feat)
        return img_rec
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)
        return optimizer
    
    def calc_loss(self, img_ori, feat):
        
        # Prepare data
        img_rec = self(feat)
        
        # Calculate loss and metrics
        mse_loss = self.mse(img_rec, img_ori)
        dssim_loss = self.dssim(img_rec, img_ori)
        id_loss = self.idloss(img_rec, feat)
        loss = mse_loss + self.alpha * dssim_loss + self.beta * id_loss
        
        acc = 0.0

        return {
            'loss': loss,
            'acc': acc,
            'img_rec': img_rec,
        }

    def training_step(self, batch, batch_idx):
        img_ori, feat = batch
        results = self.calc_loss(img_ori, feat)
        self.log('train_loss', results['loss'], prog_bar=True, on_step=True, sync_dist=True)
        self.log('train_acc', results['acc'], prog_bar=True, on_step=True, sync_dist=True)
        return results
    
    def validation_step(self, batch, batch_idx):
        img_ori, feat = batch
        results = self.calc_loss(img_ori, feat)
        self.log('val_loss', results['loss'], prog_bar=True, on_step=True)
        self.log('val_acc', results['acc'], prog_bar=True, on_step=True)
        return results

    def predict_step(self, batch, batch_idx):
        img_ori, feat = batch
        return self(feat)
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        
        save_dir = Path(f"output/{self.cfg.exp_name}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if batch_idx == 0:
            img_ori = batch[0]
            img_rec = outputs['img_rec']
            save_path = save_dir / f"{self.current_epoch}.png"
            draw_rows(img_ori, img_rec, n_pair=20, save_path=str(save_path))
        
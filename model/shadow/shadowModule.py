import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Callable
from pathlib import Path
from dataclasses import asdict

from .reconstructor import ReconstructModel
from .losses import DSSIMLoss, IDLoss
from configs import Config



class ShadowModule(pl.LightningModule):

    def __init__(self, cfg: Config, backbone:nn.Module, callback: Callable):

        super().__init__()
        self.save_hyperparameters(asdict(cfg))

        self.cfg = cfg
        self.backbone = backbone
        self.callback = callback
        self.shadow = ReconstructModel(cfg.output_size)

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.mse = nn.MSELoss()
        self.dssim = DSSIMLoss()
        self.idloss = IDLoss(self.backbone)
        
        self.lr = cfg.lr
        self.alpha = cfg.dssim_weight
        self.beta = cfg.id_weight

    def forward(self, feat):
        feat = feat.unsqueeze(-1).unsqueeze(-1)
        img_rec = self.shadow(feat)
        return img_rec
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.shadow.parameters(), lr=self.lr)
        return optimizer
    
    def calc_loss(self, img_ori, feat):
        
        # Prepare data
        img_rec = self(feat)
        
        # Calculate loss and metrics
        mse_loss = self.mse(img_rec, img_ori)
        dssim_loss = self.dssim(img_rec, img_ori)
        id_loss = self.idloss(img_rec, feat)
        loss = mse_loss + self.alpha * dssim_loss + self.beta * id_loss
        
        # Calculate cosine similarity
        feat_rec = self.backbone(img_rec)
        cosine_score = torch.nn.functional.cosine_similarity(feat, feat_rec, dim=1).mean()
        
        return {
            'loss': loss,
            'score': cosine_score,
            'img_rec': img_rec,
        }

    def training_step(self, batch, batch_idx):
        img_ori, feat = batch
        results = self.calc_loss(img_ori, feat)
        self.log('train_loss', results['loss'], prog_bar=True, sync_dist=True)
        self.log('train_score', results['score'], prog_bar=True, sync_dist=True)
        return results
    
    def validation_step(self, batch, batch_idx):
        img_ori, feat = batch
        results = self.calc_loss(img_ori, feat)
        self.log('val_loss', results['loss'], prog_bar=True)
        self.log('val_score', results['score'], prog_bar=True)
        return results

    def test_step(self, batch, batch_idx):
        img_ori, feat = batch
        results = self.calc_loss(img_ori, feat)
        self.log('test_loss', results['loss'], prog_bar=True)
        self.log('test_score', results['score'], prog_bar=True)

    def predict_step(self, batch, batch_idx):
        img_ori, feat = batch
        return self(feat)
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        
        save_dir = Path(f"output/{self.cfg.exp_name}/val")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if batch_idx == 0:
            img_ori = batch[0]
            img_rec = outputs['img_rec']
            save_path = save_dir / f"{self.current_epoch}.png"
            self.callback(img_ori, img_rec, n_pair=20, save_path=str(save_path))
        
    def on_predict_batch_end(self, outputs, batch, batch_idx):
        save_dir = Path(f"output/{self.cfg.exp_name}/pred")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        img_ori = batch[0]
        img_rec = outputs
        save_path = save_dir / f"{batch_idx}.png"
        self.callback(img_ori, img_rec, n_pair=20, save_path=str(save_path))
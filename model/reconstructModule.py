import cv2
import torch
import numpy as np
import pytorch_lightning as pl


from pathlib import Path
from datetime import datetime

from .target import TargetModel
from .shadow import ShadowModel


class ReconstructModule(pl.LightningModule):

    def __init__(self, target: TargetModel, shadow: ShadowModel, lr=1e-3):
        super(ReconstructModule, self).__init__()
        self.target = target
        self.shadow = shadow
        self.criterion = self.shadow.loss_fn
        self.lr = lr
    
    def setup(self, stage):
        self.exp_name = f"output/{datetime.now():%Y%m%d%H%M%S}"
        self.cfg_name = f"{self.target.model_type.value}+{self.shadow.model_type.value}"
        self.gpu_id = self.trainer.local_rank
        self.out_ori = []
        self.out_rec = []
        
    def forward(self, img):
        feat:torch.Tensor = self.target(img)
        feat = feat.unsqueeze(-1).unsqueeze(-1)
        return self.shadow(feat)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.shadow.parameters(), lr=self.lr)
        return optimizer
    
    def calc_loss(self, img, split:str):

        # Calculate loss
        img_rec = self(img)
        result = self.criterion(img, img_rec) # Return a dictionary with loss, score, and img_rec
        
        # Logging
        self.log(f'{split}_loss', result['loss'], prog_bar=True, sync_dist=True)
        self.log(f'{split}_score', result['score'], prog_bar=True, sync_dist=True)
        return result

    def training_step(self, batch, batch_idx):
        return self.calc_loss(batch[0], 'train') # batch[0] is the image (batch[1] is the label)
    
    def validation_step(self, batch, batch_idx):
        return self.calc_loss(batch[0], 'val')

    def test_step(self, batch, batch_idx):
        return self.calc_loss(batch[0], 'test')

    def on_save_checkpoint(self, checkpoint):
        torch.save(self.shadow.backbone.state_dict(), f"{self.cfg_name}.pt")  
    
    def visualize_batch(self, img_ori, img_rec, save_path, max_size=16):
        
        def batch2img(tensor: torch.Tensor):
            # This function converts a batch of tensor into a batch of images
            tensor = tensor.permute(0, 2, 3, 1) * 255
            ndarray = tensor.cpu().detach().numpy()
            return ndarray[:,:,:,::-1].astype(np.uint8)
        
        n = min(np.floor((len(img_ori) * 2) ** 0.5).astype(int), max_size)
        used_slice = slice(0, n**2//2)
        img_ori = batch2img(img_ori[used_slice])
        img_rec = batch2img(img_rec[used_slice])
        
        
        rows = []
        for i in range(n//2):
            start = i * n
            end = min(start + n, len(img_ori))
            row1 = np.concatenate(img_ori[start:end], axis=1)
            row2 = np.concatenate(img_rec[start:end], axis=1)
            row = np.concatenate([row1, row2], axis=0)
            rows.append(row)
            rows.append(np.zeros((10, row.shape[1], 3), dtype=np.uint8)) # the gap

        img = np.concatenate(rows, axis=0)
        cv2.imwrite(str(save_path / f"cuda{self.gpu_id}.jpg"), img)
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        # Save the original and reconstructed images in each batch
        self.out_ori.append(batch[0])
        self.out_rec.append(outputs['img_rec'])
        
    def on_validation_epoch_end(self):
        
        save_path = Path(f"{self.exp_name}(train)") / f"{self.current_epoch}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.visualize_batch(
            img_ori=torch.concat(self.out_ori), 
            img_rec=torch.concat(self.out_rec),
            save_path=save_path)

        self.out_ori.clear()
        self.out_rec.clear()

    def on_test_batch_end(self, outputs, batch, batch_idx):
        # Save the original and reconstructed images in each batch
        self.out_ori.append(batch[0])
        self.out_rec.append(outputs['img_rec'])
        
    def on_test_epoch_end(self):
            
        save_path = Path(f"{self.exp_name}(test)")
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.visualize_batch(
            img_ori=torch.concat(self.out_ori), 
            img_rec=torch.concat(self.out_rec),
            save_path=save_path)

        self.out_ori.clear()
        self.out_rec.clear()

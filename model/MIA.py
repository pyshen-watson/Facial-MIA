import math
import torch
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchvision.transforms as T

from .target import TargetModel
from .attack import AttackModel


class ModelInversionAttackModule(pl.LightningModule):

    def __init__(self, target: TargetModel, shadow: AttackModel, lr=1e-3):
        super(ModelInversionAttackModule, self).__init__()
        self.target = target
        self.shadow = shadow
        self.criterion = self.shadow.loss_fn
        self.lr = lr
    
    def setup(self, stage):
        self.exp_name = f"output/{datetime.now():%Y%m%d%H%M%S}"
        self.ckpt_name = f"{self.target.model_type.value}+{self.shadow.model_type.value}"
        self.gpu_id = self.trainer.local_rank
        
    def forward(self, img):
        return self.shadow(self.target(img))
    
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
        torch.save(self.shadow.backbone.state_dict(), f"{self.ckpt_name}.pt")  
  
    def visualize_batch(self, img_ori, img_rec, save_dir, dpi=100):
        
        # First, estimate a proper grid for visualization
        batch_size = len(img_ori)
        sample_per_row = math.ceil((batch_size / 3) ** 0.5) # Estimate the number per row, expect the grid to be a square
        grid_per_row = 3 * sample_per_row - 1 # 3 images per sample (original, reconstructed, and gap)
        num_rows = math.ceil( len(img_ori) / sample_per_row)
        
        # Secondary, calculate figsize dynamically based on image dimensions and DPI
        img_width = img_ori.shape[2] / dpi
        img_height = img_ori.shape[3] / dpi
        figsize = (img_width * grid_per_row, img_height * num_rows)
        fig, axs = plt.subplots(num_rows, grid_per_row, figsize=figsize, dpi=dpi)
        for ax in axs.flatten():
            ax.axis('off')
    
        # Convert tensor to PIL image
        to_pil = T.ToPILImage()

        for i, (img1, img2) in enumerate(zip(img_ori, img_rec)):
            
            img1 = to_pil(img1)
            img2 = to_pil(img2)
            
            row_id = i // sample_per_row
            col_id = i % sample_per_row * 3

            axs[row_id, col_id].imshow(img1)
            axs[row_id, col_id+1].imshow(img2)

        plt.tight_layout(pad=0.01)
        plt.savefig(str(save_dir / f"cuda{self.gpu_id}.jpg"))
        plt.close(fig)
 
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        
        # Only visualize the first batch
        if batch_idx != 0:
            return
        
        # Save the original and reconstructed images in each batch
        save_dir = Path(f"{self.exp_name}(val)/{self.current_epoch}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualize_batch(
            img_ori=batch[0], 
            img_rec=outputs['img_rec'],
            save_dir=save_dir)

    def on_test_batch_end(self, outputs, batch, batch_idx):
        
        # Only visualize the first batch
        if batch_idx != 0:
            return
        
        # Save the original and reconstructed images in each batch
        save_dir = Path(f"{self.exp_name}(test)")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualize_batch(
            img_ori=batch[0], 
            img_rec=outputs['img_rec'],
            save_dir=save_dir)

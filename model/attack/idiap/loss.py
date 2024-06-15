import torch.nn as nn
from torch.nn import MSELoss
from torch.nn.functional import cosine_similarity
from pytorch_msssim import SSIM
from dataclasses import dataclass, field


class DSSIMLoss(nn.Module):
    '''
    This loss calculates the DSSIM (Structural Dissimilarity) between two images.
    Lower DSSIM means the two images are more similar.
    '''
    
    def __init__(self):
        super(DSSIMLoss, self).__init__()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, img1, img2):
        dssim = (1-self.ssim(img1, img2)) / 2
        return dssim
    
class IDLoss(nn.Module):
    '''
    This loss calculates the MSE (Mean Squared Error) between two feature maps.
    '''
        
    def __init__(self):
        super(IDLoss, self).__init__()
        self.mse = MSELoss()

    def forward(self, feat1, feat2):
        return self.mse(feat1, feat2)
    

@dataclass(eq=False)
class IdiapLoss(nn.Module):
    '''
    This class integrates the DSSIM loss, ID loss and MSE loss with their weights to calculate the final loss.
    The return contains the loss, cosine similarity of two features and the reconstructed image.
    '''

    dssim_weight: float = 0.1
    id_weight: float = 0.1
    
    target: nn.Module = field(init=False)
    mse_loss: MSELoss = field(init=False, default=MSELoss())
    dssim_loss: DSSIMLoss = field(init=False, default=DSSIMLoss())
    id_loss: IDLoss = field(init=False, default=IDLoss())
    
    def __post_init__(self):
        super(IdiapLoss, self).__init__()
        
    def forward(self, img_ori, img_rec):
        
        feat_ori = self.target(img_ori)
        feat_rec = self.target(img_rec)

        MSE = self.mse_loss(img_rec, img_ori)
        DSSIM = self.dssim_loss(img_rec, img_ori)
        IDL = self.id_loss(feat_ori, feat_rec)

        loss = MSE + self.dssim_weight * DSSIM + self.id_weight * IDL
        score = cosine_similarity(feat_ori, feat_rec, dim=1).mean()

        return {
            'loss': loss,
            'score': score,
            'img_rec': img_rec
        }
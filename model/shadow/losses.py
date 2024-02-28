import torch.nn as nn
from pytorch_msssim import SSIM

class DSSIMLoss(nn.Module):
    
    def __init__(self):
        super(DSSIMLoss, self).__init__()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, img1, img2):
        dssim = (1-self.ssim(img1, img2)) / 2
        return dssim
    
class IDLoss(nn.Module):
        
    def __init__(self, backbone):
        super(IDLoss, self).__init__()
        self.backbone = backbone
        self.mse = nn.MSELoss()

    def forward(self, rec_img, ori_feat):
        rec_feat = self.backbone(rec_img)
        return self.mse(rec_feat, ori_feat)
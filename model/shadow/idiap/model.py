'''
Adapted from https://gitlab.idiap.ch/bob/bob.paper.icip2022_face_reconstruction/-/blob/a77c49644683d38f9a8b6401fbba319f0f2ad104/experiments/ArcFace/TrainNetwork/src/Network.py
Original author: H. O. Shahreza, V. K. Hahn and S. Marcel
"Face Reconstruction from Deep Facial Embeddings using a Convolutional Neural Network," 
2022 IEEE International Conference on Image Processing (ICIP), Bordeaux, France, 2022, pp. 1211-1215, doi: 10.1109/ICIP46576.2022.9897535. 
'''

import torch.nn as nn


class ReconstructBlock(nn.Module):
    
    def __init__(self, in_c, out_c, ksize, stride, padding):
        super(ReconstructBlock, self).__init__()
        
        self.dconv = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, ksize, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        
        self.skip = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )
        
    def forward(self, x):
        xd = self.dconv(x)
        xs = self.skip(xd)
        return xd + xs
    
class IdiapModel(nn.Module):

    def __init__(self, input_len: int):
        
        super(IdiapModel, self).__init__()
        self.decode1 = ReconstructBlock(input_len, 512, 4, 1, 0)
        self.decode2 = ReconstructBlock(512, 256, 4, 2, 1)
        self.decode3 = ReconstructBlock(256, 128, 4, 2, 1)
        self.decode4 = ReconstructBlock(128, 64, 4, 2, 2)
        self.decode5 = ReconstructBlock(64, 32, 4, 2, 2)
        self.decode6 = ReconstructBlock(32, 16, 4, 2, 3)
        self.final = nn.Sequential(
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.decode1(x)
        x = self.decode2(x)
        x = self.decode3(x)
        x = self.decode4(x)
        x = self.decode5(x)
        x = self.decode6(x)
        x = self.final(x)
        return x
import torch.nn as nn
from .noise_v3 import NoisyActivation
from .mobilefacenet_v3 import get_mbf_large
from .utils_v3 import discrete_cosine_transform as dct

class NoisyMobileFaceNet(nn.Module):
    
    def __init__(self, fp16, num_features, blocks=(2, 8, 12, 4), scale=4):
        super(NoisyMobileFaceNet, self).__init__()
        self.mbf = get_mbf_large(fp16, num_features, blocks, scale)
        self.noise = NoisyActivation()
        
    def forward(self, x):
        x = dct(x)
        x = self.noise(x)
        x = self.mbf(x)
        return x
    
def get_noisy_mbf_large(fp16, num_features, blocks=(2, 8, 12, 4), scale=4):
    return NoisyMobileFaceNet(fp16, num_features, blocks, scale)
import torch
import torch.nn as nn


class DecodeBlock(nn.Module):
    
    def __init__(self, in_c, out_c, ksize, stride, padding):
        super(DecodeBlock, self).__init__()
        
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
    
class Decoder(nn.Module):
    
    def __init__(self, feat_len):
        super(Decoder, self).__init__()
        
        self.decode1 = DecodeBlock(feat_len, 512, 4, 1, 0)
        self.decode2 = DecodeBlock(512, 256, 4, 2, 1)
        self.decode3 = DecodeBlock(256, 128, 4, 2, 1)
        self.decode4 = DecodeBlock(128, 64, 4, 2, 2)
        self.decode5 = DecodeBlock(64, 32, 4, 2, 2)
        self.decode6 = DecodeBlock(32, 16, 4, 2, 3)
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
        

# def test(feat_len=512, batch_size=16, device='cuda:0'):

#     feat = torch.randn(batch_size, feat_len, 1, 1).to(device)
#     model = Decoder(feat_len).to(device)
#     output = model(feat)
#     print(output.shape)


# if __name__ == '__main__':
#     test(device='cuda:0')
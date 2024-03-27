'''
Adapted from https://github.com/cavalleria/cavaface.pytorch/blob/master/backbone/mobilefacenet.py
Original author cavalleria
'''
import math
import torch.nn as nn
import torch
from torch.nn import functional as F


class DP_block(nn.Module):

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        '''
        Init method.
        '''
        super(DP_block, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, groups=groups,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_c),
            nn.GELU(),
            nn.LocalResponseNorm(size=5, alpha=5, beta=0.5, k=2)
        )
        # Differential privacy
        given_locs = torch.zeros((out_c, 56, 56))
        self.size = given_locs.shape
        mean = 0
        variance = (2 * math.log(1.25 / 1e-4)) * \
            ((1 / math.sqrt(2)) ** 2) / (3 ** 2)
        std = math.sqrt(variance)
        self.normal_noise = torch.distributions.normal.Normal(
            loc=mean, scale=std)
        '''
        variance = (2 * math.log(1.25 / 1e-5)) * (0.5 ** 2) / (3 ** 2)  # sensitivity=0.5,  delta=1e-5, epsilon=3
        math.sqrt(variance) = 0.8074675437675649 this is std
        '''

    def forward(self, x):
        '''
        Forward pass of the function.
        '''
        x = self.layers(x)
        x = x + self.DP_A()
        return x

    def DP_A(self):
        noise = self.normal_noise.sample(self.size).to(self.device)
        return noise


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, groups=groups,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_c),
            nn.PReLU(num_parameters=out_c)
        )

    def forward(self, x):
        return self.layers(x)


class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(num_features=out_c)
        )

    def forward(self, x):
        return self.layers(x)


class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(DepthWise, self).__init__()
        self.residual = residual
        self.layers = nn.Sequential(
            ConvBlock(in_c, out_c=groups, kernel=(1, 1),
                      padding=(0, 0), stride=(1, 1)),
            ConvBlock(groups, groups, groups=groups, kernel=kernel,
                      padding=padding, stride=stride),
            LinearBlock(groups, out_c, kernel=(1, 1),
                        padding=(0, 0), stride=(1, 1))
        )

    def forward(self, x):
        short_cut = None
        if self.residual:
            short_cut = x
        x = self.layers(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                DepthWise(c, c, True, kernel, stride, padding, groups))
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class GDC(nn.Module):
    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(512, 512, groups=512, kernel=(
                7, 7), stride=(1, 1), padding=(0, 0)),
            Flatten(),
            nn.Linear(512, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size))

    def forward(self, x):
        return self.layers(x)


class MobileFaceNet(nn.Module):
    def __init__(self, fp16=False, num_features=512, blocks=(1, 4, 6, 2), scale=2):
        super(MobileFaceNet, self).__init__()
        self.scale = scale
        self.fp16 = fp16
        self.layers = nn.ModuleList()
        self.layers.append(
            DP_block(3, 64 * self.scale, kernel=(3, 3),
                     stride=(2, 2), padding=(1, 1))
        )
        if blocks[0] == 1:
            self.layers.append(
                ConvBlock(64 * self.scale, 64 * self.scale, kernel=(3,
                          3), stride=(1, 1), padding=(1, 1), groups=64)
            )
        else:
            self.layers.append(
                Residual(64 * self.scale, num_block=blocks[0], groups=128, kernel=(
                    3, 3), stride=(1, 1), padding=(1, 1)),
            )

        self.layers.extend(
            [
                DepthWise(64 * self.scale, 64 * self.scale, kernel=(3, 3),
                          stride=(2, 2), padding=(1, 1), groups=128),
                Residual(64 * self.scale, num_block=blocks[1], groups=128, kernel=(
                    3, 3), stride=(1, 1), padding=(1, 1)),
                DepthWise(64 * self.scale, 128 * self.scale, kernel=(3, 3),
                          stride=(2, 2), padding=(1, 1), groups=256),
                Residual(128 * self.scale, num_block=blocks[2], groups=256, kernel=(
                    3, 3), stride=(1, 1), padding=(1, 1)),
                DepthWise(128 * self.scale, 128 * self.scale, kernel=(3, 3),
                          stride=(2, 2), padding=(1, 1), groups=512),
                Residual(128 * self.scale, num_block=blocks[3], groups=256, kernel=(
                    3, 3), stride=(1, 1), padding=(1, 1)),
            ])

        self.conv_sep = ConvBlock(
            128 * self.scale, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.features = GDC(num_features)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            for func in self.layers:
                x = func(x)
        x = self.conv_sep(x.float() if self.fp16 else x)
        x = self.features(x)
        x = F.normalize(x, p=2, dim=1)
        return x


def get_dpmbf(fp16, num_features, blocks=(1, 4, 6, 2), scale=2):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale)


def get_dpmbf_large(fp16, num_features, blocks=(2, 8, 12, 4), scale=4):
    return MobileFaceNet(fp16, num_features, blocks, scale=scale)

import torch
import torch.nn as nn

class NoisyActivation(nn.Module):
    def __init__(self, input_shape=112, budget_mean=2, sensitivity=None, require_grad=True):
        super(NoisyActivation, self).__init__()
        self.h, self.w = input_shape, input_shape
        # if sensitivity is None:
        #     sensitivity = torch.ones([189, self.h, self.w])
        # self.sensitivity = sensitivity.reshape(189 * self.h * self.w)
        self.given_locs = torch.zeros((189, self.h, self.w))
        size = self.given_locs.shape
        self.budget = budget_mean * 189 * self.h * self.w
        self.locs = nn.Parameter(torch.Tensor(size).copy_(self.given_locs))
        self.rhos = nn.Parameter(torch.zeros(size))
        self.laplace = torch.distributions.laplace.Laplace(0, 1)
        self.rhos.requires_grad = require_grad
        self.locs.requires_grad = require_grad

    def scales(self):
        softmax = nn.Softmax(dim=0)
        result = 1 / softmax(self.rhos.reshape(189 * self.h * self.w)) * self.budget
        return result.reshape(189, self.h, self.w)

    def sample_noise(self):
        epsilon = self.laplace.sample(self.rhos.shape).to(self.locs.device)
        return self.locs + self.scales() * epsilon

    def forward(self, input):
        noise = self.sample_noise()
        output = input + noise
        return output

    def aux_loss(self):
        scale = self.scales()
        loss = -1.0 * torch.log(scale.mean())
        return loss
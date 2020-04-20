import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f

from utils import ops

class Gaussian_Entropy_2D(nn.Module):
    def __init__(self, likelihood_bound=1e-9, scale_bound=1e-6):
        super(Gaussian_Entropy_2D,self).__init__()
        self.likelihood_bound = likelihood_bound
        self.scale_bound = scale_bound

    def forward(self, x, p_dec):
        # x - [N, 2*M, H, W]
        M = x.shape[1]
        mean = p_dec[:, :M, :, :]
        scale= p_dec[:, M:, :, :]

        ## to make the scale always positive
        # scale[scale == 0] = 1e-9
        scale = ops.Low_bound.apply(torch.abs(scale), self.scale_bound)

        m1 = torch.distributions.normal.Normal(mean,scale)

        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)

        likelihood = torch.abs(upper - lower)
        likelihood = ops.Low_bound.apply(likelihood, self.likelihood_bound)
        return likelihood

class Gaussian_Entropy_3D(nn.Module):
    def __init__(self, likelihood_bound=1e-9, scale_bound=1e-6):
        super(Gaussian_Entropy_3D,self).__init__()
        self.likelihood_bound = likelihood_bound
        self.scale_bound = scale_bound

    def forward(self, x, p_dec, quan_step = 1., scaler = None):

        mean = p_dec[:, 0, :, :, :]
        scale = p_dec[:, 1, :, :, :]
        if scaler == None:
            pass
        else:
            mean = scaler[0](mean, lamb = scaler[2], train = scaler[3])
            scale = scaler[1](scale, lamb = scaler[2], train = scaler[3])
        ## to make the scale always positive
        # scale[scale == 0] = 1e-9
        scale = ops.Low_bound.apply(torch.abs(scale), self.scale_bound)
        #scale1 = torch.clamp(scale1,min = 1e-9)
        m1 = torch.distributions.normal.Normal(mean,scale)
        lower = m1.cdf(x - 0.5 * quan_step)
        upper = m1.cdf(x + 0.5 * quan_step)

        likelihood = torch.abs(upper - lower)

        likelihood = ops.Low_bound.apply(likelihood, self.likelihood_bound)
        return likelihood

class Laplace_Entropy_3D(nn.Module):
    def __init__(self):
        super(Laplace_for_entropy,self).__init__()

    def forward(self, x, p_dec):
        mean = p_dec[:, 0,:, :, :]
        scale= p_dec[:, 1,:, :, :]

        ## to make the scale always positive
        scale[scale == 0] = 1e-9
        #scale1 = torch.clamp(scale1,min = 1e-6)
        m1 = torch.distributions.laplace.Laplace(mean,scale)

        lower = m1.cdf(x - 0.5)
        upper = m1.cdf(x + 0.5)
        likelihood = torch.abs(upper - lower)

        likelihood = ops.Low_bound.apply(likelihood,1e-6)
        return likelihood
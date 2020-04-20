import torch.nn as nn
import torch
import math
from utils.ops import GDN
from torch.autograd import Variable
import torch.nn.functional as f
import numpy as np

class ResGDN(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,padding,inv=False):
        super(ResGDN,self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.inv = bool(inv)
        self.conv1 = nn.Conv2d(self.in_ch,self.out_ch,self.k, self.stride
                                             ,self.padding)
        self.conv2 = nn.Conv2d(self.in_ch,self.out_ch,self.k, self.stride
                                             ,self.padding)
        self.ac1 = GDN(self.in_ch,self.inv)
        self.ac2 = GDN(self.in_ch,self.inv)

    def forward(self,x):
        x1 = self.ac1(self.conv1(x))
        x2 = self.conv2(x1)
        out = self.ac2(x + x2)
        return out

class ResBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,padding):
        super(ResBlock,self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, self.k, self.stride
                               , self.padding)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch, self.k, self.stride
                               , self.padding)

    def forward(self,x):
        x1 = self.conv2(f.relu(self.conv1(x)))
        out = x+x1
        return out

# here use embedded gaussian
class Non_local_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Non_local_Block,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g = nn.Conv2d(self.in_channel,self.out_channel, 1, 1, 0)
        self.theta = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.phi = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.W = nn.Conv2d(self.out_channel, self.in_channel, 1, 1, 0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self,x):
        # x_size: (b c h w)

        batch_size = x.size(0)
        
        theta_x = self.theta(x).view(batch_size,self.out_channel,-1)
        theta_x = theta_x.permute(0,2,1)

        phi_x = self.phi(x).view(batch_size, self.out_channel, -1)

        g_x = self.g(x).view(batch_size,self.out_channel,-1)
        g_x = g_x.permute(0,2,1)
        
        # TODO: sparse NLAM
        
        f1 = torch.matmul(theta_x,phi_x)
        f_div_C = f.softmax(f1,dim=-1)
        y = torch.matmul(f_div_C,g_x)
        y = y.permute(0,2,1).contiguous()
        y = y.view(batch_size,self.out_channel,*x.size()[2:])
        W_y = self.W(y)
        z = W_y+x

        return z

class CConv2D(nn.Module):
    def __init__(self, num_lamb = None, channels_in = None, conv=True, kernel = 3, groups = 1):
        super(CConv2D,self).__init__()
        # self.fc_scale = nn.Linear(num_lamb, channels_in)
        # self.fc_bias = nn.Linear(num_lamb, channels_in)
        self.num_lamb = num_lamb
        self.channels_in = channels_in
        self.with_conv = conv
        x = math.log(math.exp(1.)-1)
        self.u = nn.Parameter(x*torch.ones([channels_in,1,num_lamb]))
        self.v = nn.Parameter(torch.zeros([channels_in,1,num_lamb]))
        if self.with_conv:
            self.conv = nn.Conv2d(channels_in, channels_in, kernel, 1, kernel//2, groups=groups)

    def interp(self, x, lamb):
        # print(lamb)
        f1 = int(lamb)
        f2 = int(lamb + 1)

        # print(f1,f2)
        ratio = lamb - np.floor(lamb)
        x_onehot_floor = f.one_hot(torch.LongTensor([f1]), num_classes=self.num_lamb)
        x_onehot_ceil = f.one_hot(torch.LongTensor([f2]), num_classes=self.num_lamb)
        x_onehot = (1. - ratio) * x_onehot_floor.type(dtype=torch.float32) + ratio * x_onehot_ceil.type(dtype=torch.float32)
        x_onehot = x_onehot.repeat(self.channels_in,1,1).view(self.channels_in, -1, 1)

        return x_onehot.cuda()

    def forward(self, x, lamb, train):
        if train:
            x_onehot = f.one_hot(lamb, num_classes=self.num_lamb)
            x_onehot = x_onehot.type(dtype=torch.float32)
            x_onehot = x_onehot.repeat(self.channels_in,1,1).view(self.channels_in, -1, 1)
        else:
            x_onehot = self.interp(x,lamb)
        # print(x_onehot.size())
        scale = torch.matmul(self.u, x_onehot)
        bias = torch.matmul(self.v, x_onehot)

        # print(x_onehot.size(), scale.size(), bias.size(), x.size())
        if self.with_conv:
            y = f.softplus(scale) * self.conv(x) + bias
        else:
            y = f.softplus(scale) * x + bias
        return y

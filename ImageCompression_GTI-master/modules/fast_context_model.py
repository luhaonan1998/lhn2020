import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.basic_module import ResBlock
from modules.gaussian_entropy_model import Gaussian_Entropy_3D

class MaskConv3d(nn.Conv3d):
    def __init__(self, mask_type,in_ch, out_ch, kernel_size, stride, padding):
        super(MaskConv3d, self).__init__(in_ch, out_ch, kernel_size, stride, padding,bias=True)

        self.mask_type = mask_type
        ch_out, ch_in, k, k, k = self.weight.size()
        mask = torch.zeros(ch_out, ch_in, k, k, k)
        central_id = k*k*(k//2)+k*(k//2)
        central_id2 = k*k*(k//2)+k*(k//2+1)
        current_id = 1
        if mask_type=='A':
            for i in range(k):
                for j in range(k):
                    for t in range(k):
                        if current_id <= central_id:
                            mask[:, :, i, j, t] = 1
                        else:
                            mask[:, :, i, j, t] = 0
                        current_id = current_id + 1

        if mask_type=='B':
            for i in range(k):
                for j in range(k):
                    for t in range(k):
                        if current_id <= central_id2:
                            mask[:, :, i, j, t] = 1
                        else:
                            mask[:, :, i, j, t] = 0
                        current_id = current_id + 1

        self.register_buffer('mask', mask)
    def forward(self, x):

        self.weight.data *= self.mask
        return super(MaskConv3d,self).forward(x)

class Context(nn.Module):
    def __init__(self):
        super(Context, self).__init__()
        self.conv1 = MaskConv3d('A', 1, 24, 5, 1, 2)

        self.conv2 = nn.Sequential(nn.Conv3d(25,48,1,1,0),nn.LeakyReLU(),nn.Conv3d(48,96,1,1,0),nn.LeakyReLU(),
                                   nn.Conv3d(96,2,1,1,0))
        self.conv3 = nn.Conv2d(384,192,3,1,1)

        self.gaussin_entropy_func = Distribution_for_entropy2()


    def forward(self, x,hyper):
        x = torch.unsqueeze(x, dim=1)
        hyper = torch.unsqueeze(self.conv3(hyper),dim=1)
        x1 = self.conv1(x)

        output = self.conv2(torch.cat((x1,hyper),dim=1))
        p = self.gaussin_entropy_func(torch.squeeze(x,dim=1),output)

        return p,output

class Context2(nn.Module):
    def __init__(self):
        super(Context2, self).__init__()
        self.conv1 = nn.Sequential(MaskConv3d('A', 1, 24, 5, 1, 2),nn.LeakyReLU())
        self.conv1_1 = nn.Sequential(MaskConv3d('B',24,24,5,1,2),nn.LeakyReLU())
        self.conv1_2 = MaskConv3d('B',24,24,5,1,2)
        self.conv2 = nn.Sequential(nn.Conv3d(25,48,1,1,0),nn.LeakyReLU(),nn.Conv3d(48,96,1,1,0),nn.LeakyReLU(),
                                   nn.Conv3d(96,2,1,1,0))
        self.conv3 = nn.Conv2d(128,64,3,1,1)

        self.gaussian_entropy_func = Gaussian_Entropy_3D()


    def forward(self, x,hyper):
        x = torch.unsqueeze(x, dim=1)
        hyper = torch.unsqueeze(self.conv3(hyper),dim=1)
        x1 = self.conv1(x)
        x11 = self.conv1_1(x1)
        x12 = self.conv1_2(x11)
        x2 = x12+x1
        output = self.conv2(torch.cat((x2,hyper),dim=1))
        p3 = self.gaussin_entropy_func(torch.squeeze(x,dim=1),output)

        return p3,output

class Context3(nn.Module):
    def __init__(self):
        super(Context3, self).__init__()
        self.conv1 = nn.Sequential(MaskConv3d('A', 1, 24, 5, 1, 2),nn.LeakyReLU())
        self.conv1_1 = nn.Sequential(MaskConv3d('B',24,24,5,1,2),nn.LeakyReLU())
        self.conv1_2 = MaskConv3d('B',24,24,5,1,2)
        self.conv2 = nn.Sequential(nn.Conv3d(25,48,1,1,0),nn.LeakyReLU(),nn.Conv3d(48,96,1,1,0),nn.LeakyReLU(),
                                   nn.Conv3d(96,2,1,1,0))
        self.conv3 = nn.Conv2d(384,192,3,1,1)

        self.gaussin_entropy_func = Distribution_for_entropy2()


    def forward(self, x,hyper):
        x = torch.unsqueeze(x, dim=1)
        hyper = torch.unsqueeze(self.conv3(hyper),dim=1)
        x1 = self.conv1(x)
        x11 = self.conv1_1(x1)
        x12 = self.conv1_2(x11)
        x2 = x12+x1
        output = self.conv2(torch.cat((x2,hyper),dim=1))
        p3 = self.gaussin_entropy_func(torch.squeeze(x,dim=1),output)

        return p3,output

class Context4(nn.Module):
    def __init__(self, M):
        super(Context4, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(2*M,M,3,1,1),nn.LeakyReLU())
        self.mask_conv1 = MaskConv3d('A', 1, 24, 5, 1, 2)
        self.mask_conv2 = nn.Sequential(nn.Conv3d(25,64,1,1,0),nn.LeakyReLU(),nn.Conv3d(64,96,1,1,0),nn.LeakyReLU(),
                                   nn.Conv3d(96,2,1,1,0))
        
        self.gaussian_entropy_func = Gaussian_Entropy_3D()

    def forward(self, x, hyper, quan_step = 1., scaler = None):
        # x: main_encoder's quantized output
        # hyper: hypder_decoder's output
        x = torch.unsqueeze(x, dim=1)
        
        hyper = torch.unsqueeze(self.conv1(hyper),dim=1)
        x1 = self.mask_conv1(x)
        output = self.mask_conv2(torch.cat((x1,hyper),dim=1))
        
        p = self.gaussian_entropy_func(torch.squeeze(x,dim=1), output, quan_step, scaler = scaler)
        
        return p, output
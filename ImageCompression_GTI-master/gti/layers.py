'''
All Rights Reserved.

Copyright (c) 2017-2019, Gyrfalcon technology Inc.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gti.quantize as Q
from gti.chip import spec

"""Refactored convolution and relu
boolean tensors are uint8 before pytorch 1.2
bias must always be present to appease the optimizer"""
class Conv2d(nn.Conv2d):
    def __init__(self, quant_params=None, mask_bit=None, *args, **kwargs):
        if quant_params.quant_w and (quant_params.chip is None or mask_bit is None):
            raise ValueError("Must specify chip and mask bit when quantizing.")
        super(Conv2d, self).__init__(*args, **kwargs)
        self.register_buffer("quantize", torch.tensor(quant_params.quant_w))
        self.chip = quant_params.chip #string so not easily saved; ASCII is dirty and messes up the dicts
        self.register_buffer("mask_bit", torch.tensor(mask_bit))
        self.bias.requires_grad = quant_params.fuse
    
    def forward(self, x):
        if self.quantize:
            shift = Q.compute_shift(self.weight, self.bias, self.chip, self.mask_bit.item()).item()
            tmp_weight = Q.quantize_weight(self.weight, self.mask_bit, shift)
            if self.bias is not None: #TODO(Yin): is this right???
                tmp_bias = Q.QuantizeShift.apply(self.bias, shift)
            else:
                tmp_bias = None
            return F.conv2d(x, tmp_weight, tmp_bias, self.stride, self.padding, self.dilation, self.groups)
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class ReLU(nn.ReLU):
    def __init__(self, quantize=False, cap=31.0, act_bits=5, **kwargs):
        super(ReLU, self).__init__(**kwargs)
        self.register_buffer("quantize", torch.tensor(quantize))
        self.register_buffer("cap", torch.tensor(cap, dtype=torch.float32))
        self.register_buffer("max_act", torch.tensor(spec.MAX_ACTIVATION_VALUE[act_bits]))
    
    def forward(self, x):
        if self.quantize:
            out = 0.5 * (torch.abs(x) - torch.abs(x - self.cap) + self.cap)
            factor = (self.max_act / self.cap)#.item() #uses less GPU RAM
            return Q.Round.apply(out * factor) / factor
        return F.relu(x, inplace=self.inplace)

class Upsample(nn.Module):
    def __init__(self, num_channels, upsampling_fill_mode=spec.UpSamplingFillMode.REPEAT):
        super(Upsample, self).__init__()
        self.num_channels = num_channels
        if upsampling_fill_mode == spec.UpSamplingFillMode.REPEAT:
            self.register_buffer("up", torch.ones(num_channels, 1, 2, 2))
        elif upsampling_fill_mode == spec.UpSamplingFillMode.ZERO:
            self.register_buffer("up", torch.zeros(num_channels, 1, 2, 2))
            self.up[:,0,0,0]=1
        else:
            raise ValueError("Invalid upsampling_fill_mode")

    def forward(self, x):
        return F.conv_transpose2d(x, self.up,
                                stride=2, groups=self.num_channels)

"""Computation block"""
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                padding=1, dilation=1, groups=1, quant_params=None, mask_bit=1, cal=False):
        super(conv_block, self).__init__()
        self.conv = Conv2d(quant_params, mask_bit,
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = ReLU(quant_params.quant_act, 0, 5, inplace=True) #TODO figure out way to pass param correctly
        self.register_buffer("fuse", torch.tensor(quant_params.fuse))
        self.cal = cal #only done once, no need to save var

    def set_status(self, qw, qa, fuse, cal=None):
        self.relu.quantize[...] = 1 if qa else 0
        self.conv.quantize[...] = 1 if qw else 0
        self.fuse[...] = 1 if fuse else 0
        self.conv.bias.requires_grad = fuse
        if cal is not None:
            self.cal = cal

    def forward(self, x):
        x = self.conv(x)
        if not self.fuse:
            x = self.bn(x)
        x = self.relu(x)
        if self.cal:
            #finds max (over epoch) of 99th percentile of each batch
            y = x.cpu().detach().numpy()
            temp = np.percentile(y, 99)
            if temp > self.relu.cap.item():
                self.relu.cap[...] = temp
        return x

"""Higher order Computation blocks
    all are either nn.Sequential or subclass it"""
def make_basic_block(in_channels, out_channels, quant_params,
        mask_bits, size, cal,
        padding=1, pool=True, finalChans=None):
    block1 = conv_block(in_channels, out_channels,
            kernel_size=3, padding=padding, quant_params=quant_params, mask_bit=mask_bits, cal=cal)
    block2 = conv_block(out_channels, out_channels,
            kernel_size=3, padding=padding, quant_params=quant_params, mask_bit=mask_bits, cal=cal)
    layers = [block1, block2]
    if size>2:
        if finalChans is None:
            finalChans=out_channels
        layers.append(conv_block(out_channels, finalChans,
            kernel_size=3, padding=padding, quant_params=quant_params, mask_bit=mask_bits, cal=cal))
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

#due to residual connection, in channels must = out channels
#vanilla resnet has a conv in the identity path, which allows changing channel #
#we do not have that option
def make_residual_block(channels, quant_params, mask_bits, cal, padding=1, pool=True):
    block1 = conv_block(channels, channels,
            kernel_size=3, padding=padding, quant_params=quant_params, mask_bit=mask_bits, cal=cal)
    block2 = conv_block(channels, channels,
            kernel_size=3, padding=padding, quant_params=quant_params, mask_bit=mask_bits, cal=cal)
    layers = [block1, block2]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return ResBlock(*layers)

#Residual block with similar interface to nn.Seq
#hard coded to only accept 2 blocks + optional pool
class ResBlock(nn.Sequential):
    def __init__(self, *args):
        super(ResBlock, self).__init__(*args)
        if type(args[-1])==nn.MaxPool2d:
            self.pool = True
        else:
            self.pool = False
        
    #TODO: generalize to more than 2-3 blocks later?
    def forward(self, x):
        identity = x
        out = self[0](x)
        block2 = self[1]
        out = block2.conv(out)
        if not block2.fuse:
            out = block2.bn(out)
        out += identity
        out = block2.relu(out)
        if block2.cal:
            y = out.cpu().detach().numpy()
            temp = np.percentile(y, 99)
            if temp > block2.relu.cap.item():
                block2.relu.cap.data[...] = temp
        if self.pool:
            out = self[2](out)
        return out

def make_single_conv_block(in_channels, out_channels, stride, quant_params, mask_bit, cal):
    return nn.Sequential(
        conv_block(in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, quant_params=quant_params, mask_bit=mask_bit, cal=cal),
    )

def make_depthwise_sep_block(in_channels, out_channels, stride, quant_params, mask_bit, cal):
    return nn.Sequential(
        conv_block(in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, quant_params=quant_params, mask_bit=mask_bit, cal=cal),
        conv_block(in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, groups=1, quant_params=quant_params, mask_bit=mask_bit, cal=cal)
    )

def make_deconv_block(in_channels, out_channels, quant_params, mask_bits, cal,
                    upsampling_fill_mode=spec.UpSamplingFillMode.REPEAT):
    """GTI device supported 'deconvolution', i.e. upsampling followed by GTI conv2d
    
    upsampling fill mode, see spec.UpSamplingFillMode:
        - REPEAT: fill with repeats of current value, for example:
            1 becomes [1, 1]
                      [1, 1]
        - ZERO: fill with zeros, for example:
            1 becomes [1, 0]
                      [0, 0]
    """
    return nn.Sequential(Upsample(in_channels, upsampling_fill_mode),
                        conv_block(in_channels, out_channels,
                                quant_params=quant_params, mask_bit=mask_bits, cal=cal)
    )

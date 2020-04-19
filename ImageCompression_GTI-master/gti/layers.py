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

"""Basic layers
boolean tensors are uint8 before pytorch 1.2"""
class Conv2d(nn.Conv2d):
    """A layer that implements 2D convolutions.
        Bias must always be present to appease the optimizer
        Bias is 0 initialized by default
        It can get a grad only if quant_params.fuse is True
        Consequently, it can only be not 0 if fuse is True

    Args:
        quant_params: GTI quantization parameters
        mask_bit: mask bitwidth for GTI quantized convolution
        *args/**kwargs - see nn.Conv2d
    """
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
            shift = Q.compute_shift(
                self.weight,
                self.bias if self.bias.requires_grad else None,
                self.chip,
                self.mask_bit.item()
            ).item()
            tmp_weight = Q.quantize_weight(
                 self.weight,
                 self.mask_bit,
                 shift
            )
            if self.bias.requires_grad: #TODO(Yin): is this right???
                tmp_bias = Q.QuantizeShift.apply(self.bias, shift)
            else:
                tmp_bias = None
            return F.conv2d(
                       x,
                       tmp_weight,
                       tmp_bias,
                       self.stride,
                       self.padding,
                       self.dilation,
                       self.groups
                   )
        return F.conv2d(
                   x,
                   self.weight,
                   self.bias,
                   self.stride,
                   self.padding,
                   self.dilation,
                   self.groups
               )

class ReLU(nn.ReLU):
    """A layer that implements ReLU.

    Args:
        quantize: if True, quantizes activations
        cap: when quantization is active, activations are clipped to be no
            larger than cap
        act_bits: number of bits of precision to represent the activations
        **kwargs - see nn.ReLU
    """
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
    """A layer that implements Upsampling.

    Args:
        in_channels: number of input channels
        upsampling_mode: see models/deconv.py for a description of what the
            different modes mean
    """
    def __init__(self, in_channels, upsampling_mode="REPEAT"):
        super(Upsample, self).__init__()
        self.in_channels = in_channels
        if upsampling_mode == "REPEAT":
            self.register_buffer("up", torch.ones(in_channels, 1, 2, 2))
        elif upsampling_mode == "ZERO":
            self.register_buffer("up", torch.zeros(in_channels, 1, 2, 2))
            self.up[:,0,0,0]=1
        else:
            raise ValueError("Invalid upsampling_mode: " + upsampling_mode)

    def forward(self, x):
        return F.conv_transpose2d(
                   x,
                   self.up,
                   stride=2,
                   groups=self.in_channels
               )

class Flatten(nn.Module):
    """A layer that implements Flatten.
        Only necessary before pytorch 1.2

    Args:
        None
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Stride2Pooling(nn.Module):
    """A layer that implements Stride2Pooling.
        Also called TopLeftPooling/SamplePooling (in each 2x2 block, picks
            top left)
        Only needed for ResBlock because residual addition occurs
        after conv, but before the stride 2

    Args:
        None
    """
    def __init__(self):
        super(Stride2Pooling, self).__init__()

    def forward(self, x):
        return x[:,:,::2,::2]

"""Main computation block"""
class ConvBlock(nn.Module):
    """The main computation block that consists of a conv, (optional) BN, relu.

    Args:
        use_bn: if True, this block includes BN
        quant_params: GTI quantization parameters
        mask_bit: mask bitwidth for GTI quantized convolution
        other args: see nn.Conv2d

    Arg not included in interface:
        stride=1 by default
        stride=2 is a special case, and handled by the higher order computation blocks
        stride>2 not supported by chip
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                padding=1, dilation=1, groups=1, use_bn=True, quant_params=None, mask_bit=1):
        super(ConvBlock, self).__init__()
        self.register_buffer("use_bn", torch.tensor(use_bn))
        self.register_buffer("fuse", torch.tensor(quant_params.fuse))
        self.cal = False #only done once, no need to save var
        new_fuse = quant_params.fuse or not use_bn
        old_fuse = quant_params.fuse
        quant_params.fuse = new_fuse
        self.conv = Conv2d(
            quant_params,
            mask_bit,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        quant_params.fuse = old_fuse
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if quant_params.ten_bit_act:
            act_bits = 10
        else:
            act_bits = 5
        self.relu = ReLU(
            quant_params.quant_act,
            0,
            act_bits,
            inplace=True
        )

    def set_status(self, qw, qa, fuse, cal=None):
        self.relu.quantize[...] = qa
        self.conv.quantize[...] = qw
        self.fuse[...] = fuse
        self.conv.bias.requires_grad = fuse
        if cal is not None:
            self.cal = cal

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn and not self.fuse:
            x = self.bn(x)
        x = self.relu(x)
        if self.cal:
            #finds max (over epoch) of 99th percentile of each batch
            y = x.cpu().detach().numpy()
            temp = np.percentile(y, 99)
            if temp > self.relu.cap.item():
                self.relu.cap[...] = temp
        return x

class ReLUWrapper(nn.Module):
    """Wraps a GTI ReLU layer to behave like a computation block.
        This is only used for the RELU_AFTER_ADDITION mode of residual blocks.

    Args:
        see ReLU
    """
    def __init__(self, quantize=False, cap=31.0, act_bits=5, **kwargs):
        super(ReLUWrapper, self).__init__()
        self.relu = ReLU(quantize=False, cap=31.0, act_bits=5, **kwargs)
        self.cal = False

    def set_status(self, qw, qa, fuse, cal=None):
        self.relu.quantize[...] = qa
        if cal is not None:
            self.cal = cal

    def forward(self, x):
        x = self.relu(x)
        if self.cal:
            #finds max (over epoch) of 99th percentile of each batch
            y = x.cpu().detach().numpy()
            temp = np.percentile(y, 99)
            if temp > self.relu.cap.item():
                self.relu.cap[...] = temp
        return x

"""Higher order computation blocks: all are either nn.Sequential or subclass it"""
def basic_conv_block(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        groups=1,
        downsample_mode="MAXPOOL",
        block_size=1,
        use_bn=True,
        quant_params=None,
        mask_bit=1
    ):
    """Basic convolution blocks comprised of multiple GTI conv layers.

    Args:
        in_channels: input channels, fixed for each conv layer except the first layer
        out_channels: output channels, fixed for all conv layers
        kernel_size: kernel size, fixed for all conv layers
        padding: has to be 1 for all conv layers due to chip compatibility
        downsample_mode: None = no downsampling; "MAXPOOL" = 2x2, stride 2
            Maxpool at the end; "STRIDE2" = stride 2 on last conv layer.
            On any given chip, "MAXPOOL" and "STRIDE2" cannot be mixed,
            but cannot be checked here.  A similar constraint applies
            to upsampling_mode in deconv_block(s), if any.
        block_size: number of convolution layers
        use_bn: if True, every conv is followed by a BN (then relu)
        quant_params: GTI quantization parameters
        mask_bit: mask bitwidth for GTI quantized convolution
        groups: used for group convolutions - see nn.Conv2d

    Returns:
        GTI basic conv blocks wrapped in nn.sequential layers
    """

    assert block_size>0, "at least 1 block required"
    layers = []
    for _ in range(block_size):
        layers.append(
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                use_bn=use_bn,
                quant_params=quant_params,
                mask_bit=mask_bit
            )
        )
        in_channels = out_channels
    if downsample_mode:
        if downsample_mode=="MAXPOOL":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif downsample_mode=="STRIDE2":
            layers[-1].conv.stride = (2,2)
        else:
            raise NotImplementedError("Unrecognized downsample_mode: " + downsample_mode)
    return nn.Sequential(*layers)

class ResBlock(nn.Sequential):
    """Residual block with similar interface to nn.Seq
        This is necessary because nn.Seq cannot implement residual connections.

    Args:
        resnet_mode: see residual_block below
        *args: see nn.Sequential
    """
    def __init__(self, *args, resnet_mode="ORIGINAL"):
        super(ResBlock, self).__init__(*args)
        if resnet_mode=="ORIGINAL":
            if len(args)%2:
                self.pool = True
            else:
                self.pool = False
        elif resnet_mode=="RELU_AFTER_ADDITION":
            mod = len(args)%3
            if mod==1:
                self.pool = True
            elif mod==0:
                self.pool = False
            else:
                print(args)
                assert False, "Improperly made ResBlock"
            self.forward = self.forward2
        else:
            raise NotImplementedError("Resnet mode not supported: " + resnet_mode)

    def forward(self, x):
        '''conv block, conv, optional BN, residual add, relu
            There may be an optional pool at the end of this block.'''
        for block_idx in range(0,len(self)-1,2):
            identity = x
            x = self[block_idx](x)
            block2 = self[block_idx+1]
            x = block2.conv(x)
            if block2.use_bn and not block2.fuse:
                x = block2.bn(x)
            x += identity
            x = block2.relu(x)
            if block2.cal:
                y = x.cpu().detach().numpy()
                temp = np.percentile(y, 99)
                if temp > block2.relu.cap.item():
                    block2.relu.cap.data[...] = temp
        if self.pool:
            x = self[-1](x)
        return x

    def forward2(self, x):
        '''conv block, conv block, residual add, relu
            There may be an optional pool at the end of this block.'''
        for block_idx in range(0,len(self)-1,3):
            identity = x
            x = self[block_idx](x)
            x = self[block_idx+1](x)
            x += identity
            x = self[block_idx+2](x) #relu
        if self.pool:
            x = self[-1](x)
        return x

def residual_block(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        downsample_mode="MAXPOOL",
        block_size=1,
        use_bn=True,
        quant_params=None,
        mask_bit=1,
        resnet_mode="ORIGINAL"
    ):
    """Residual blocks comprised of multiple residual convolution pairs.
        Due to residual connection, in channels must = out channels.
        Vanilla resnet has a conv in the identity path, which allows
            changing channel #; we do not have that option

    Args:
        in_channels: input channels, fixed for each conv layer except the first layer
        out_channels: output channels, fixed for all conv layers
        kernel_size: kernel size, fixed for all conv layers
        padding: has to be 1 for all conv layers due to chip compatibility
        downsample_mode: None = no downsampling; "MAXPOOL" = 2x2, stride 2
            Maxpool at the end; "STRIDE2" = stride 2 on last conv layer.
            On any given chip, "MAXPOOL" and "STRIDE2" cannot be mixed,
            but cannot be checked here.  A similar constraint applies
            to upsampling_mode in deconv_block(s), if any.
        block_size: number of convolution layers
        use_bn: if True, every conv is followed by a BN (then relu)
        quant_params: GTI quantization parameters
        mask_bit: mask bitwidth for GTI quantized convolution
        resnet_mode: "ORIGINAL" - residual blocks consist of: ConvBlock, conv,
            BN, add in identity, relu.  See also arxiv:1603.05027 figure 4a.
            "RELU_AFTER_ADDITION" - residual blocks consist of: ConvBlock,
            ConvBlock, add in identity, (another) relu.  See also
            arxiv:1603.05027 figure 4c, but with an extra relu at the end.
            Due to chip limitation, residual blocks always appear in pairs.

    Returns:
        GTI residual conv blocks (may contains multiple residual paris \
        depending on block_size) wrapped in nn.sequential layers
    """

    assert in_channels==out_channels, "input channels must equal \
        output channels for GTI resnet block!"
    assert block_size>0, "at least 1 block required"
    assert resnet_mode in ["ORIGINAL", "RELU_AFTER_ADDITION"]
    if quant_params.ten_bit_act:
        act_bits = 10
    else:
        act_bits = 5
    layers = []
    for block_idx in range(block_size*2):
        layers.append(
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                use_bn=use_bn,
                quant_params=quant_params,
                mask_bit=mask_bit
            )
        )
        if resnet_mode!="ORIGINAL" and block_idx%2:
            layers.append(
                ReLUWrapper(
                    quant_params.quant_act,
                    0,
                    act_bits,
                    inplace=True
                )
            )
    if downsample_mode:
        if downsample_mode=="MAXPOOL":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif downsample_mode=="STRIDE2":
            layers.append(Stride2Pooling())
        else:
            raise NotImplementedError("Unrecognized downsample_mode: " + downsample_mode)
    return ResBlock(*layers, resnet_mode=resnet_mode)

def depthwise_sep_block(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        downsample_mode=None,
        block_size=1,
        use_bn=True,
        quant_params=None,
        mask_bit=1
    ):
    """Depthwise separable blocks comprised of multiple pairs of pointwise conv + depthwise conv.
        IMPORTANT: the order here is opposite to the commonly accepted order.
        This ordering reflects a chip limitation: mask_bit can only change
        between major layers, and cannot change within a major layer.
        Since a stride 2 conv ends a major layer, it would be awkward for this
        block to start with a stride 2 conv.
        To recover a sequence of "regular" depthwise separable blocks, pad the
        beginning and end with basic_conv_blocks.

    Args:
        in_channels: input channels, fixed for each conv layer except the first layer
        out_channels: output channels, fixed for all conv layers
        kernel_size: predefined; variable ignored
        padding: has to be 1 for all conv layers due to chip compatibility
        downsample_mode: None = no downsampling; "MAXPOOL" = 2x2, stride 2
            Maxpool at the end; "STRIDE2" = stride 2 on last conv layer.
            On any given chip, "MAXPOOL" and "STRIDE2" cannot be mixed,
            but cannot be checked here.  A similar constraint applies
            to upsampling_mode in deconv_block(s), if any.
        block_size: number of convolution layers
        use_bn: if True, every conv is followed by a BN (then relu)
        quant_params: GTI quantization parameters
        mask_bit: mask bitwidth for GTI quantized convolution

    Returns:
        GTI depthwise separable blocks (may contain multiple pointwise
        and depthwise pairs depending on block_size) wrapped in nn.Sequential
    """
    assert block_size>0, "at least 1 block required"
    layers = []
    for _ in range(block_size):
        layers.append(
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                groups=1,
                use_bn=use_bn,
                quant_params=quant_params,
                mask_bit=mask_bit
            )
        )
        in_channels = out_channels
        layers.append(
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=padding,
                groups=in_channels,
                use_bn=use_bn,
                quant_params=quant_params,
                mask_bit=mask_bit
            )
        )
    if downsample_mode:
        if downsample_mode=="MAXPOOL":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif downsample_mode=="STRIDE2":
            layers[-1].conv.stride = (2,2)
        else:
            raise NotImplementedError("Unrecognized downsample_mode: " + downsample_mode)
    return nn.Sequential(*layers)

def deconv_block(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        upsampling_mode="REPEAT",
        block_size=1,
        use_bn=True,
        quant_params=None,
        mask_bit=1
    ):

    """GTI device supported 'deconvolution', i.e. upsampling followed by GTI conv layers

    Args:
        in_channels: input channels, fixed for each conv layer except the first layer
        out_channels: output channels, fixed for all conv layers
        kernel_size: kernel size, fixed for all conv layers
        padding: has to be 1 for all conv layers due to chip compatibility
        upsampling_mode:
            - REPEAT: fill with repeats of current value, for example:
                1 becomes [1, 1]
                          [1, 1]
            - ZERO: fill with zeros, for example:
                1 becomes [1, 0]
                          [0, 0]
            On any given chip, "REPEAT" and "ZERO" cannot be mixed,
            but cannot be checked here.  A similar constraint applies
            to downsample_mode in the other block(s), if any.
        block_size: number of convolution layers
        use_bn: if True, every conv is followed by a BN (then relu)
        quant_params: GTI quantization parameters
        mask_bit: mask bitwidth for GTI quantized convolution

    Returns:
        GTI deconv blocks (may contains multiple conv layers \
        depending on block_size) wrapped in nn.sequential layers
    """

    assert in_channels==out_channels, "input channel must equal \
        output channel for GTI deconv block!"
    assert block_size>0, "at least 1 block required"

    layers = [Upsample(in_channels, upsampling_mode)]
    for _ in range(block_size):
        layers.append(
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                use_bn=use_bn,
                quant_params=quant_params,
                mask_bit=mask_bit
            )
        )
    return nn.Sequential(*layers)

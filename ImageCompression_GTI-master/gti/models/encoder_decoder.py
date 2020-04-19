import torch
import torch.nn as nn
import gti.chip.spec as spec
from gti.layers import (
    basic_conv_block,
    deconv_block,
    residual_block
)
# 4 downsample in main & 2 downsample in hyper

# All host parts run on PC use 12-bit weight quantization
# For deconv, zero padding upsampling is used, refer to gti/layers.py for details 
host_mask_bit = 12
_UPSAMPLING_MODE = "ZERO" # previous version uses repeat sampling

"""
  Please note that network-to-chip mapping is based on GTI 2803 chip constraints
  For other chips, we would have other mapping methods
"""

"""
  Main encoder is split into host0 + chip0 + host1
  host0: take 8-bit image data and output 5-bit feature maps
         put it on host PC, because chip can only take 5 bit as input
  host1: take 5-bit feature maps and output floating-point feature map (not QuantReLU)
         put it on host PC, becasue chip can not support non-ReLU conv.
"""
def make_main_encoder_host0(args):
    return nn.Sequential(
        basic_conv_block(3, 256, downsample_mode=None,
            block_size=1, use_bn=False, quant_params=args, mask_bit=host_mask_bit)
    )

def make_main_encoder_chip0(args):
    mask_bits = spec.specs[args.chip]['main_encoder']
    return nn.Sequential(
        basic_conv_block(256, 256, downsample_mode="STRIDE2",
            block_size=1, use_bn=False, quant_params=args, mask_bit=mask_bits[0]),

        residual_block(256, 256, downsample_mode=None,
            block_size=3, use_bn=False, quant_params=args, mask_bit=mask_bits[1]),
        basic_conv_block(256, 256, downsample_mode="STRIDE2",
            block_size=2, use_bn=False, quant_params=args, mask_bit=mask_bits[1]),

        residual_block(256, 256, downsample_mode=None,
            block_size=3, use_bn=False, quant_params=args, mask_bit=mask_bits[2]),
        basic_conv_block(256, 256, downsample_mode="STRIDE2",
            block_size=1, use_bn=False, quant_params=args, mask_bit=mask_bits[2]),

        residual_block(256, 256, downsample_mode=None,
            block_size=3, use_bn=False, quant_params=args, mask_bit=mask_bits[2]),
    )

def make_main_encoder_host1(args):
    return nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),    # No ReLU with stride 2
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)    # No ReLU with stride 1
    )


"""
  Hyper encoder is split into host0 + chip0 + host1
  host0: take floating-point feature maps and output 5-bit feature map
  host1: take 5-bit feature maps and output floating-point feature map
         it consists of 3 conv layers and the last conv. does not have ReLU
"""
def make_hyper_encoder_host0(args):
    return nn.Sequential(
        basic_conv_block(256, 256, downsample_mode=None,
            block_size=1, use_bn=False, quant_params=args, mask_bit=host_mask_bit) 
    )
def make_hyper_encoder_chip0(args):
    mask_bits = spec.specs[args.chip]['hyper_encoder']
    return nn.Sequential(
        residual_block(256, 256, downsample_mode=None,
            block_size=3, use_bn=False, quant_params=args, mask_bit=mask_bits[0]),
        basic_conv_block(256, 256, downsample_mode="STRIDE2",
            block_size=2, use_bn=False, quant_params=args, mask_bit=mask_bits[0]),

        residual_block(256, 256, downsample_mode=None,
            block_size=3, use_bn=False, quant_params=args, mask_bit=mask_bits[1]),
        basic_conv_block(256, 256, downsample_mode="STRIDE2",
            block_size=2, use_bn=False, quant_params=args, mask_bit=mask_bits[1]),

    )

def make_hyper_encoder_host1(args):
    return nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)    # No ReLU
    )


"""
  Hyper decoder is split into host0 + chip0 + host1
  host0: take floating-point feature maps and output 5-bit feature maps 
         it consists of 8 7x7 conv. layers 
         put on PC because GTI 2803 can not support 7x7 input 
  host1: take 5-bit feature maps and output floating-point feature map
         put it on host PC, becasue chip can not support non-ReLU conv.
"""
def make_hyper_decoder_host0(args):
    return nn.Sequential(
        residual_block(256, 256, downsample_mode=None,
            block_size=3, use_bn=False, quant_params=args, mask_bit=host_mask_bit),
        #basic_conv_block(256, 256, downsample_mode=None,
        #    block_size=1, use_bn=False, quant_params=args, mask_bit=host_mask_bit),

        deconv_block(256, 256, upsampling_mode=_UPSAMPLING_MODE,
            block_size=1, use_bn=False, quant_params=args, mask_bit=host_mask_bit),
        basic_conv_block(256, 256, downsample_mode=None,
           block_size=1, use_bn=False, quant_params=args, mask_bit=host_mask_bit),

        deconv_block(256, 256, upsampling_mode=_UPSAMPLING_MODE,
            block_size=1, use_bn=False, quant_params=args, mask_bit=host_mask_bit),        
    )

def make_hyper_decoder_chip0(args):
    mask_bits = spec.specs[args.chip]['hyper_decoder']
    return nn.Sequential(
        residual_block(256, 256, downsample_mode=None,
            block_size=3, use_bn=False, quant_params=args, mask_bit=mask_bits[0])
    )

def make_hyper_decoder_host1(args):
    return nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)    # No ReLU
    )



"""
  Main decoder is split into host0 + chip0 + host1
  host0: take floating-point feature maps and output 5-bit feature map
  host1: take 5-bit feature maps and output floating-point feature map
         it consists of 3 conv layers and the last conv. does not have ReLU
"""
def make_main_decoder_host0(args):
    return nn.Sequential(
        basic_conv_block(256, 256, downsample_mode=None,
            block_size=1, use_bn=False, quant_params=args, mask_bit=host_mask_bit)
    )

def make_main_decoder_chip0(args):
    mask_bits = spec.specs[args.chip]['main_decoder']
    return nn.Sequential(
        basic_conv_block(256, 256, downsample_mode=None,
            block_size=1, use_bn=False, quant_params=args, mask_bit=mask_bits[0]),
        deconv_block(256, 256, upsampling_mode=_UPSAMPLING_MODE,
            block_size=1, use_bn=False, quant_params=args, mask_bit=mask_bits[0]),
        
        residual_block(256, 256, downsample_mode=None,
            block_size=3, use_bn=False, quant_params=args, mask_bit=mask_bits[1]),
        #basic_conv_block(256, 256, downsample_mode=None,
        #    block_size=1, use_bn=False, quant_params=args, mask_bit=mask_bits[1]),
        deconv_block(256, 256, upsampling_mode=_UPSAMPLING_MODE,
            block_size=1, use_bn=False, quant_params=args, mask_bit=mask_bits[1]),

        residual_block(256, 256, downsample_mode=None,
            block_size=3, use_bn=False, quant_params=args, mask_bit=mask_bits[2]),
        #basic_conv_block(256, 256, downsample_mode=None,
        #    block_size=3, use_bn=False, quant_params=args, mask_bit=mask_bits[2]),
        deconv_block(256, 256, upsampling_mode=_UPSAMPLING_MODE,
            block_size=1, use_bn=False, quant_params=args, mask_bit=mask_bits[2]),
        
        residual_block(256, 256, downsample_mode=None,
            block_size=3, use_bn=False, quant_params=args, mask_bit=mask_bits[2]),
        #basic_conv_block(256, 256, downsample_mode=None,
        #    block_size=3, use_bn=False, quant_params=args, mask_bit=mask_bits[2]),
        deconv_block(256, 256, upsampling_mode=_UPSAMPLING_MODE,
            block_size=1, use_bn=False, quant_params=args, mask_bit=mask_bits[2])

    )

def make_main_decoder_host1(args):
    return nn.Sequential(
        nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)    # No ReLU
    )

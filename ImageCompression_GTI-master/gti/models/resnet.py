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

import torch.nn as nn
import torch
from gti.layers import make_basic_block, make_residual_block
import gti.chip.spec as spec
from gti.models.gtinet import GtiNet
import numpy as np

def make_layers(args):
    cal = False
    mask_bits = spec.specs[args.chip]['resnet18']
    return nn.Sequential(
        make_basic_block(3, 64, args, mask_bits[0], 2, cal),
        make_basic_block(64, 64, args, mask_bits[1], 2, cal),
        make_residual_block(64, args, mask_bits[2], cal, pool=False),
        make_residual_block(64, args, mask_bits[3], cal),
        make_basic_block(64, 128, args, mask_bits[4], 2, cal, pool=False),
        make_residual_block(128, args, mask_bits[5], cal),
        make_basic_block(128, 256, args, mask_bits[6], 2, cal, pool=False),
        make_residual_block(256, args, mask_bits[7], cal),
        make_basic_block(256, 512, args, mask_bits[8], 2, cal, pool=False),
        make_residual_block(512, args, mask_bits[9], cal, pool=False),
        nn.AdaptiveAvgPool2d((1, 1))
    )

#unlike VGG, sometimes the first few layers are never quantized ->
#set status not defined here
class ResNetLike(GtiNet):
    def __init__(self):
        super(ResNetLike, self).__init__()

    @staticmethod
    def get_status_checkpoint(checkpoint):
        qa = checkpoint['module.chip_layer.2.0.relu.quantize']
        qa = True if qa else False
        qw = checkpoint['module.chip_layer.2.0.conv.quantize']
        qw = True if qw else False
        fuse = checkpoint['module.chip_layer.2.0.fuse']
        fuse = True if fuse else False
        return qw, qa, fuse

class resnet18(ResNetLike):
    def __init__(self, args):
        super(resnet18, self).__init__()
        self.chip_layer = make_layers(args)
        self.host_layer = nn.Linear(512, args.num_classes)
        self._initialize_weights()

    @staticmethod
    def get_num_classes(checkpoint):
        try:
            return checkpoint['module.host_layer.weight'].shape[0]
        except KeyError:
            return -1

    def forward(self, x):
        x = self.chip_layer(x).view(x.size(0), -1)
        return self.host_layer(x)
     
    def set_status(self, qw, qa, fuse, cal=None):
        for major in self.chip_layer:
            if type(major)==nn.AdaptiveAvgPool2d:
                break
            for minor in major:
                if type(minor) is nn.MaxPool2d:
                    continue
                minor.set_status(qw, qa, fuse, cal)

    #mode=0: dump the mismatched vars
    def modify_num_classes(self, checkpoint, mode=0):
        change_flag = False
        current_num_classes = self.host_layer.weight.shape[0]
        incoming_num_classes = resnet18.get_num_classes(checkpoint)
        if incoming_num_classes != current_num_classes:
            change_flag = True
            keys_to_be_deleted = [
                'module.host_layer.weight',
                'module.host_layer.bias'
            ]
            if mode!=0:
                raise NotImplementedError("change num classes mode != 0 not yet implemented")
            for key in keys_to_be_deleted:
                try:
                    del checkpoint[key]
                except KeyError:
                    pass
        return checkpoint, change_flag
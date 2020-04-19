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
from gti.layers import make_basic_block
import gti.chip.spec as spec
from gti.models.gtinet import GtiNet

#creates major layers 0-5
def make_gnetfc_layers(args):
    cal = False
    mask_bits = spec.specs[args.chip]['gnetfc']
    return nn.Sequential(
        make_basic_block(3, 64, args, mask_bits[0], 2, cal),
        make_basic_block(64, 128, args, mask_bits[1], 2, cal),
        make_basic_block(128, 256, args, mask_bits[2], 3, cal),
        make_basic_block(256, 256, args, mask_bits[3], 3, cal),
        make_basic_block(256, 256, args, mask_bits[4], 3, cal),
        make_basic_block(256, 256, args, mask_bits[5], 3, cal, 0, False, args.num_classes)
    )

#creates major layers 0-4
def make_vgg_layers(args):
    cal = False
    mask_bits = spec.specs[args.chip]['vgg16']
    return nn.Sequential(
            make_basic_block(3, 64, args, mask_bits[0], 2, cal),
            make_basic_block(64, 128, args, mask_bits[1], 2, cal),
            make_basic_block(128, 256, args, mask_bits[2], 3, cal),
            make_basic_block(256, 512, args, mask_bits[3], 3, cal),
            make_basic_block(512, 512, args, mask_bits[4], 3, cal)
    )

class VggLike(GtiNet):
    def __init__(self):
        super(VggLike, self).__init__()

    #returns tuple of bools
    #assume dictionary is for net wrapped in nn.DataParallel
    @staticmethod
    def get_status_checkpoint(checkpoint):
        qa = checkpoint['module.chip_layer.0.0.relu.quantize']
        qa = True if qa else False
        qw = checkpoint['module.chip_layer.0.0.conv.quantize']
        qw = True if qw else False
        fuse = checkpoint['module.chip_layer.0.0.fuse']
        fuse = True if fuse else False
        return qw, qa, fuse

    #needed because loading checkpoint overrides these vars
    def set_status(self, qw, qa, fuse, cal=None):
        for major in self.chip_layer:
            for minor in major:
                if type(minor) is nn.MaxPool2d:
                    continue
                minor.set_status(qw, qa, fuse, cal)

class StumpTest(VggLike):
    def __init__(self, args):
        super(StumpTest, self).__init__()
        self.chip_layer = nn.Sequential(
            make_basic_block(3, 64, args, 3, 2, args.cal),
            make_basic_block(64, 128, args, 3, 3, args.cal, 0, False, args.num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.chip_layer(x).view(x.size(0), -1)

class gnetfc(VggLike):
    def __init__(self, args):
        super(gnetfc, self).__init__()
        self.chip_layer = make_gnetfc_layers(args)
        self._initialize_weights()

    @staticmethod
    def get_num_classes(checkpoint):
        try:
            return checkpoint['module.chip_layer.5.2.conv.weight'].shape[0]
        except KeyError:
            return -1

    def forward(self, x):
        return self.chip_layer(x).view(x.size(0), -1)

    #mode=0: dump the mismatched vars
    def modify_num_classes(self, checkpoint, mode=0):
        change_flag = False
        current_num_classes = self.chip_layer[5][2].conv.weight.shape[0]
        incoming_num_classes = gnetfc.get_num_classes(checkpoint)
        if incoming_num_classes != current_num_classes:
            change_flag = True
            keys_to_be_deleted = [
                'module.chip_layer.5.2.conv.weight',
                'module.chip_layer.5.2.conv.bias'
            ]
            for key in checkpoint.keys():
                if "5.2.bn" in key:
                    keys_to_be_deleted.append(key)
            if mode!=0:
                raise NotImplementedError("change num classes mode != 0 not yet implemented")
            for key in keys_to_be_deleted:
                try:
                    del checkpoint[key]
                except KeyError:
                    pass
        return checkpoint, change_flag

class vgg16(VggLike):
    def __init__(self, args):
        super(vgg16, self).__init__()
        self.chip_layer = make_vgg_layers(args)
        self.host_layer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, args.num_classes),
        )
        self._initialize_weights()

    @staticmethod
    def get_num_classes(checkpoint):
        try:
            return checkpoint['module.host_layer.6.weight'].shape[0]
        except KeyError:
            return -1

    def forward(self, x):
        x = self.chip_layer(x).view(x.size(0), -1)
        return self.host_layer(x)

    #mode=0: dump the mismatched vars
    def modify_num_classes(self, checkpoint, mode=0):
        change_flag = False
        current_num_classes = self.host_layer[6].weight.shape[0]
        incoming_num_classes = vgg16.get_num_classes(checkpoint)
        if incoming_num_classes != current_num_classes:
            change_flag = True
            keys_to_be_deleted = [
                'module.host_layer.6.weight',
                'module.host_layer.6.bias'
            ]
            # for key in checkpoint.keys():
            #     if "5.2.bn" in key:
            #         keys_to_be_deleted.append(key)
            if mode!=0:
                raise NotImplementedError("change num classes mode != 0 not yet implemented")
            for key in keys_to_be_deleted:
                try:
                    del checkpoint[key]
                except KeyError:
                    pass
        return checkpoint, change_flag

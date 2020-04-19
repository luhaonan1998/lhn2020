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
import gti.layers

#abstract base class
class GtiNet(nn.Module):
    def __init__(self):
        super(GtiNet, self).__init__()

    @staticmethod
    def get_status_checkpoint(checkpoint):
        raise NotImplementedError

    @staticmethod
    def get_num_classes(checkpoint):
        raise NotImplementedError

    def set_status(self, qw, qa, fuse, cal=None):
        raise NotImplementedError

    def modify_num_classes(self, checkpoint, mode=0):
        raise NotImplementedError

    #BN merge must be done before gain edit
    #if done together, BN is automatically done first
    def fuse(self, do_fuse=True, do_gain=True):
        prev_cap = 31.0
        for m in self.modules(): #recursively goes through every module and submodule
            if isinstance(m, gti.layers.conv_block):
                if do_fuse: 
                    gamma = m.bn.weight
                    beta = m.bn.bias
                    mean = m.bn.running_mean
                    var = m.bn.running_var
                    bn_epsilon = 1e-6
                    bn_stddev = torch.sqrt(var + bn_epsilon)
                    bn_factor = gamma / bn_stddev
                    for i in range(bn_factor.shape[0]):
                        m.conv.weight.data[i] *= bn_factor[i]
                    m.conv.bias = nn.Parameter(beta - bn_factor * mean)
                    #nn.Param has req_grad=True by default
                    
                if do_gain:
                    this_cap = m.relu.cap.item()

                    b_gain = 31.0 / this_cap
                    m.conv.bias.data *= b_gain

                    w_gain = prev_cap / this_cap
                    m.conv.weight.data *= w_gain

                    prev_cap = this_cap
                    m.relu.cap.data[...] = 31.0
            #assuming all FC layers come after all conv layers
            #assuming 1st encountered FC layer is also 1st FC layer executed by net
            if type(m)==nn.Linear:
                if do_gain:
                    m.weight.data *= prev_cap/31.0
                break
        return prev_cap

    #should be same for all classes
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

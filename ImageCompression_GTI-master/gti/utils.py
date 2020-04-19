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

"""Model util functions."""

import os
import sys
import logging
import subprocess
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from importlib import import_module
from gti.param_parser import QuantizationParams
_logger = logging.getLogger(__name__)

def train_epoch(net, criterion, optimizer, train_loader):
    """run training over one epoch"""
    torch.set_grad_enabled(True)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        outputs = net(inputs)
        targets = targets.to(outputs.device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().item()

    train_loss/=(batch_idx+1)
    acc = 100.*float(correct)/float(total)
    if train_loss<1e-3:
        _logger.info('Training loss: %.4e | accuracy: %.3f%% (%d/%d)'
            % (train_loss, acc, correct, total))
    else:
        _logger.info('Training loss: %.4f | accuracy: %.3f%% (%d/%d)'
            % (train_loss, acc, correct, total))
    return acc, train_loss

def val_epoch(net, criterion, val_loader):
    """run evaluation over one epoch"""
    torch.set_grad_enabled(False)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        outputs = net(inputs)
        targets = targets.to(outputs.device)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().item()

    test_loss/=(batch_idx+1)
    acc = 100.*float(correct)/float(total)
    _logger.info('Validation loss: %.4f | accuracy: %.3f%% (%d/%d)'
            % (test_loss, acc, correct, total))
    return acc, test_loss


#checkpoint related operations
# TODO: check GPU/CPU compatibility 
def load_checkpoint(chip, net, checkpoint, use_gpu=True):
    """load checkpoint and construct net given chip, net name and checkpoint"""
    _logger.info('Loading checkpoint.. %s' % checkpoint)

    device = "cuda" if use_gpu else "cpu"
    state_dict = torch.load(
        checkpoint, 
        map_location=device
    )
    state_dict = state_dict['model_state_dict']

    arch = get_architecture(net)
    num_classes = arch.get_num_classes(state_dict)
    train_args = wrap_args(
        chip = chip,
        quant_w = True,
        quant_act = True,
        fuse = True,
        num_classes = num_classes
    )
    net = arch(train_args)
    net = nn.DataParallel(net, [0])  # TODO: if this is OK for CPU
    if not use_gpu:
        torch.cuda.is_available = return_false
        net.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)
        net.cuda()
        cudnn.benchmark = True 
    # fix batchnorm, dropout and others
    net.eval() 
    return net

def save_checkpoint(save_file, epoch, net, optimizer, best_acc):
    """Save checkpoint to a given file path"""
    _logger.info('Saving checkpoint to %s' % save_file)
    state = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(state, save_file)

def check_consistency(netKeys, ckptKeys):
    netKeys = set(netKeys)
    ckptKeys = set(ckptKeys)
    missing = 0
    extra = 0
    for key in netKeys:
        if key in ckptKeys:
            ckptKeys.discard(key)
        else:
            _logger.warning(key + " is missing!")
            missing+=1
    for key in ckptKeys:
        _logger.warning(key + " is discarded!")
        extra+=1
    if missing>0 or extra>0:
        _logger.warning(
            "missing: {}, extra: {}"
            .format(
                missing, 
                extra
            )
        )

def get_step(quant_w, quant_act, fuse):
    """Get which step according to quant_w, quant_act and fuse"""
    if not quant_w and not quant_act and not fuse: 
        return 1
    elif quant_w and not quant_act and not fuse: 
        return 2
    elif quant_w and quant_act and not fuse:
        return 3
    elif quant_w and quant_act and fuse:
        return 4
    else:
        _logger.warning(
            "Checkpoint not originated from standard GTI MDK training flow! \
            Proceed training with quant_w: %s, quant_act: %s, and \
            fuse: %s" %(quant_w, quant_act, fuse)
        )
        return -1 

def get_checkpoint_name(args):
    """Given args (which specifies quantization schemes & chip),
    returns default name to save checkpoint to"""
    
    prefix = "%s_%s"%(args.chip, args.net)
    step = get_step(args.quant_w, args.quant_act, args.fuse)
    if args.resume_from or step in [-1, 1]:
       resume_from = args.resume_from
    else: 
       resume_from = os.path.join(
             args.best_checkpoint_dir, 
             prefix + "_step%s.pt"%(str(step-1))
        ) 
    save_to = prefix + "_step%s.pt"%(str(step))
    return resume_from, save_to

def get_architecture(arch):
    """Get model architecture based on model name"""
    arch=arch.lower() #arch is str
    if arch in ['gnetfc', 'vgg16']:
        module = import_module("gti.models.vgg")
    elif "resnet" in arch:
        module = import_module("gti.models.resnet")
    elif arch == "mobilenet":
        module = import_module("gti.models.mobilenet")
    else:
        raise NotImplementedError(arch)
    return eval("module."+arch)


def train_step_msg(args):
    """Tag checkpoint to indicate quantization schemes & chip"""
    step = get_step(args.quant_w, args.quant_act, args.fuse)
    if step == 1: 
        train_msg = "Step1-training floating-point model"
    elif step == 2:
        train_msg = "Step2-training weight-quantized model"
    elif step == 3:
        train_msg = "Step3-training activation-quantized model"
    elif step == 4:
        train_msg = "Step4-Finetuning fully-quantized model"
    else:
        train_msg = "Step unknow-GTI training flow not followed"
    return train_msg


def wrap_args(chip, quant_w, quant_act, fuse, num_classes=1000):
    """Wrap training-related parameters into wrap"""
    args = QuantizationParams().parse_args([])
    args.chip = chip
    args.quant_w = quant_w
    args.quant_act = quant_act
    args.fuse = fuse
    args.num_classes = num_classes
    return args

def save_labels_txt(sorted_labels, file_name):
    with open(file_name, "w") as f:
        for idx, label in enumerate(sorted_labels):
            f.write(str(idx) + " " + label + "\n")

def get_sorted_classes(folder_name):
    class_list = [d for d in os.listdir(folder_name)
        if os.path.isdir(os.path.join(folder_name, d))
        and any(os.scandir(os.path.join(folder_name, d)))]
    class_list.sort()
    return class_list

#used for preventing torch from seeing GPUs
def return_false():
    return False


def make_call_with(log_fname):
    def call(s):
        with open(log_fname, 'a') as f:
            f.write(s + '\n')
        if subprocess.call(s, shell=True) != 0:
            print("Error occurred during processing. Check log file {} for error messages.".format(log_fname))
            sys.exit(1)
    return call

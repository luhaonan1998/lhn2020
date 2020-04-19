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

"""Convert Pytorch checkpoint to chip format."""

import json
import os
import shutil
import numpy as np
import torch
from pathlib import Path

import logging
# When on-chip model performance is not satisfactory, you may set _DEBUG_CONVERSION to True to see
# more details during conversion by setting environment variable before running conversion script:
#   GTI_DEBUG_CONVERSION=True python convert_to_chip.py
_DEBUG_CONVERSION = os.environ.get("GTI_DEBUG_CONVERSION") == "True"
_CONVERSION_LOG_LEVEL = logging.DEBUG if _DEBUG_CONVERSION else logging.INFO
_logger = logging.getLogger(__name__)
_logger.setLevel(_CONVERSION_LOG_LEVEL)

from gti.chip import driver
from gti.config import gticonfig
from gti.utils import get_step
import gti.quantize as Q

# GNetFC specific handling: need to pad last layer channels of conv weights and biases to 256
_GNETFC_CHIP_OUT_CHANNELS = 256
_MODEL_STATE_DICT_KEY = "model_state_dict"

def convert(checkpoint, net, dat_json, model_json, labels_txt, out_model, evaluate_path, debug):
    """Convert checkpoint to chip-compatible .model

    Args:
        checkpoint (str): path of checkpoint, e.g. checkpoints/best/2801_step1.pt
        net (str): type of net corresponding to checkpoint
        dat_json (str): path of DAT definition JSON
        model_json (str): path of MODEL definition JSON
        labels_txt (str): path of labels.txt file containing mapping between index and label name
        out_model (str): path of output model to be generated
        evaluate_path (str): 
    Returns:
        None. Generate output model and write to disk.
    """
    debug_dir = os.path.join(os.path.dirname(out_model), "debug")
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)

    state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)[_MODEL_STATE_DICT_KEY]
    data_files = convert_chip_layers(
        state_dict=state_dict,
        net=net,
        dat_json=dat_json,
        save_dir=debug_dir,
        evaluate_path=evaluate_path
    ) #creates chip.dat, fills in data_files dictionary, updates dat.json

    #TODO: clean up hard coding
    update_num_classes = ()
    if "gnetfc" not in net:
        convert_host_layers(state_dict, data_files, debug_dir)
        if "vgg" in net:
            num_classes = state_dict['module.host_layer.6.weight'].shape[0]
            update_num_classes = ('fc8', num_classes)
        else:
            num_classes = state_dict['module.host_layer.weight'].shape[0]
            update_num_classes = ('fc', num_classes)
    data_files["label"] = os.path.realpath(labels_txt)
    update_model_json(model_json, data_files, update_num_classes)

    if os.path.exists(out_model):
        _logger.warning("{} already exists and will be overwritten".format(out_model))
    driver.compose_model(json_file=model_json, model_file=out_model)
    if not debug:
        shutil.rmtree(debug_dir)
    _logger.info("successfully generated {}".format(out_model))

def convert_chip_layers(state_dict, net, dat_json, save_dir, evaluate_path):
    """Convert chip layers into .DAT file

    Args:
        state_dict (dict): model state dictionary of checkpoint
        dat_json (str): path of DAT definition JSON
        save_dir (str): directory to save intermediate files
        net (str): type of net
    Obsolete/may be useful in future:
        activation_bits (int): number of bits for activation on-chip for last layer

    Returns:
        dictionary containing paths to data files, look up by key
        data files consist of:
            "dat0", "filter", "bias"
        This function also generates these files as a side effect
    """
    is_gnetfc = False
    if net == "gnetfc":
        is_gnetfc = True #special handling for gnetfc (padding for final layer)
        
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)
    filter_file = os.path.join(save_dir, "filter.txt")
    bias_file = os.path.join(save_dir, "bias.txt")
    dat_out = os.path.join(save_dir, "chip.dat")

    #get filter, bias, shifts
    flat_filter = np.array([])
    flat_bias = np.array([])
    bit_shifts = []

    with open(dat_json, "r") as f:
        j = json.load(f)
    chip = str(j["model"][0]["ChipType"])
    #TODO: try to pass in this var?

    # major_layers = len(j["layer"])
    # for mjr_idx, mjr_layer in enumerate(j["layer"]):
    #     sub_layers = mjr_layer["sublayer_number"]
    #     for sub_idx in range(sub_layers):
    mjr_idx = -1
    while True:
        mjr_idx+=1
        sub_idx=-1
        while True:
            sub_idx+=1
            try:
                weight = state_dict["module.chip_layer.{}.{}.conv.weight".format(mjr_idx, sub_idx)]
            except KeyError:
                _logger.debug("pytorch major layer {} has {} sublayers".format(mjr_idx, sub_idx))
                break
            bias = state_dict["module.chip_layer.{}.{}.conv.bias".format(mjr_idx, sub_idx)]
            mask_bit = state_dict["module.chip_layer.{}.{}.conv.mask_bit".format(mjr_idx, sub_idx)].item()
            # mask_bit = j["layer"][mjr_idx]["coef_bits"]
            shift = Q.compute_shift(
                weight=weight,
                bias=bias,
                chip=chip,
                mask_bit=mask_bit
            )
            weight = Q.quantize_weight(weight=weight, mask_bit=mask_bit, shift=shift)
            bias = Q.QuantizeShift.apply(bias, shift)
            weight = weight.detach().numpy()
            bias = bias.detach().numpy()

            # gnetfc special handling
            # pad the last convolutional layer weight and bias output channels with 0
            # is_last = (mjr_idx == major_layers - 1 and sub_idx == sub_layers - 1)
            is_last = (mjr_idx == 5 and sub_idx == 2)
            if is_gnetfc and is_last:
                weight = np.pad(
                    array=weight,
                    pad_width=(
                        (0, _GNETFC_CHIP_OUT_CHANNELS - weight.shape[0]),
                        (0, 0),
                        (0, 0),
                        (0, 0)
                    ),
                    mode="constant"
                )
                bias = np.pad(
                    array=bias,
                    pad_width=(0, _GNETFC_CHIP_OUT_CHANNELS - bias.shape[0]),
                    mode="constant"
                )

            # Log detailed information for layer gains and parameter magnitudes
            _logger.debug("Layer: {}_{}".format(mjr_idx + 1, sub_idx + 1))
            _logger.debug(
                "|W|max: {}, |B|max: {}, Shift: {}"
                .format(np.amax(np.absolute(weight)), np.amax(np.absolute(bias)), shift)
            )
            _logger.debug("")
            flat_filter = np.concatenate((flat_filter, weight.ravel()))
            flat_bias = np.concatenate((flat_bias, bias.ravel()))
            bit_shifts.append(shift)
        if sub_idx==0:
            break #no sublayer -> empty major layer -> nothing left
    _logger.info("converting convolutional layers to .DAT file")
    flat_filter.tofile(filter_file, sep="\n", format="%.16e")
    flat_bias.tofile(bias_file, sep="\n", format="%.16e")
    
    dat_json_out = os.path.join(save_dir, os.path.basename(dat_json))
    update_dat_json(dat_json=dat_json, new_shifts=bit_shifts, dat_json_out=dat_json_out, evaluate_path=evaluate_path)

    #now that dat_json if updated, filter/bias files are written to disk
    #write chip.dat to disk @ dat_out
    # print("b4 config")
    gticonfig(
        dat_json=dat_json_out,
        filter_file=filter_file,
        bias_file=bias_file,
        dat_out=dat_out,
        save_dir=save_dir
    )
    # print("after config")
    return {
        "dat0": os.path.realpath(dat_out),
        "filter": os.path.realpath(filter_file),
        "bias": os.path.realpath(bias_file)
    }

#run after convert_chip_layers
#adds entries to existing data_files and returns it
#currently only needed for vgg16/resnet18/mobilenet
def convert_host_layers(state_dict, data_files, save_dir):
    fc_locations = []
    for key in state_dict:
        if "module.host_layer" in key and "weight" in key:
            key = key.split(".")
            if len(key)==3:
                break
            fc_locations.append(key[2])
    if len(fc_locations)>0:
        #multiple FC layers -> vgg16
        for gti_idx, idx in enumerate(fc_locations):
            weight = state_dict["module.host_layer.{}.weight".format(idx)]
            bias = state_dict["module.host_layer.{}.bias".format(idx)]
            #probably for 1st 7x7 conv layer?
            # if idx == 0 and reshape_to is not None:
            #         w = w.reshape(reshape_to)
            #         w = w.transpose(get_permute_axes("HWIO", "OIHW"))
            #         in_size = np.prod(reshape_to[:3])
            #         out_size = reshape_to[3]
            #         w = w.reshape((out_size, in_size))
            bin_path = os.path.join(save_dir, "fc{}.bin".format(gti_idx+6)) #TODO: unhardcode later
            with open(bin_path, "wb") as f:
                out_size, in_size = weight.shape
                np.array([in_size], dtype="<i").tofile(f)
                np.array([out_size], dtype="<i").tofile(f)
                weight = np.asarray(weight, order="C")
                weight.tofile(f)
                bias = np.asarray(bias)
                bias.tofile(f)
            data_files["fc{}".format(gti_idx+6)] = os.path.realpath(bin_path)
    elif "module.host_layer.weight" in state_dict:
        #Resnet18 or MN
        weight = state_dict["module.host_layer.weight"]
        bias = state_dict["module.host_layer.bias"]
        bin_path = os.path.join(save_dir, "fc.bin")
        with open(bin_path, "wb") as f:
            out_size, in_size = weight.shape
            np.array([in_size], dtype="<i").tofile(f)
            np.array([out_size], dtype="<i").tofile(f)
            weight = np.asarray(weight, order="C")
            weight.tofile(f)
            bias = np.asarray(bias)
            bias.tofile(f)
        data_files["fc"] = os.path.realpath(bin_path)
    return data_files

#does not touch the other vars -> assumes they're already correct
def update_model_json(model_json, data_files, update_num_classes=()):
    """Update full MODEL JSON with newly generated data file paths:
        dat0, dat1... (chip layers)
        fc (host layers), labels.txt

    Args:
        model_json (str): path of DAT definition JSON
        data_files (dict str:str): name:file path
        update_num_classes (tuple str,int): layer name, # of classes (ie outputs)

    Returns:
        None"""
    with open(model_json, "r+") as f:
        model_def = json.load(f)
        count_dat = 0
        for layer in model_def["layer"]:
            if layer["operation"] == "GTICNN":
                layer["data file"] = data_files["dat" + str(count_dat)]
                count_dat += 1
            elif layer["operation"] == "LABEL":
                layer["data file"] = data_files["label"]
            elif layer["operation"] == "FC":
                layer["data file"] = data_files[layer["name"]]
                if update_num_classes and layer['name']==update_num_classes[0]:
                    layer["output channels"] = update_num_classes[1]
            elif layer["operation"] == "SOFTMAX" and update_num_classes:
                layer["output channels"] = update_num_classes[1]
        f.seek(0)
        json.dump(model_def, f, indent=4, sort_keys=True)
        f.truncate()

#does not touch the other vars -> assumes they're already correct
#most of the other vars can be easily read/computed from the checkpoint
#image_size for each layer can be computed (knowing input size), but is annoying
#pooling information is not in the checkpoint
def update_dat_json(dat_json, new_shifts, dat_json_out, evaluate_path):
    """Update DAT JSON with newly calculated bit shifts/scaling factors from checkpoint.
    
    Args:
        dat_json (str): path of DAT definition JSON
        new_shifts (list(int)): list of new shifts

    Returns:
        None"""

    with open(dat_json) as f:
        net_config = json.load(f)
    
    # add MajorLayerNumber
    net_config['model'][0]['MajorLayerNumber'] = len(net_config['layer'])

    # add major_layer and shift values to net.json
    idx = 0
    for i, layer in enumerate(net_config['layer']):
        layer['major_layer'] = i + 1
        layer['scaling'] = []
        for i in range(layer['sublayer_number']):
            layer['scaling'].append(int(new_shifts[idx]))
            idx += 1

    #change net.json learning mode to do the conversion
    if evaluate_path is not None:
        if os.path.isdir(evaluate_path): # need turn off all the learning mode
            for layer in net_config['layer']:
                if 'learning' in layer and layer['learning']:
                    layer['learning'] = False          
        elif os.path.isfile(evaluate_path):  # need turn on all the learning mode
            for layer in net_config['layer']:
                if 'learning' not in layer or not layer['learning']:
                    print("try to turn on learning")
                    layer['learning'] = True

    with open(dat_json_out, 'w') as f:
        json.dump(net_config, f, indent=4, separators=(',', ': '), sort_keys=True)






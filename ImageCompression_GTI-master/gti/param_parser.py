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

"""Argument parsers to parse training & evaluation parameters passed as script arguments."""

import argparse
from .chip import spec

class BaseParams(argparse.ArgumentParser):
    def __init__(self):
        super(BaseParams, self).__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.add_argument(
            "--net",
            type=str,
            default="vgg16",
            help="currently supported networks: gnetfc, vgg16, mobilenet, resnet18"
        ) 
        self.add_argument(
            "--chip",
            type=str,
            default="2801",
            help="target chip, e.g. 2801, 2803, 5801"
        )
        self.add_argument(
            "--num_classes",
            type=int,
            default=1000,
            help="number of output classes"
        )
        self.add_argument(
            "--image_size",
            type=int,
            default=spec.DEFAULT_IMAGE_SIZE,
            choices=spec.ALLOWED_IMAGE_SIZES,
            help="image size, where size == height == width"
        )
        self.add_argument(
            "--checkpoint",
            type=str,
            default="checkpoints/best/2801_vgg16_step4.pt",
            help="checkpoint path",
        )

class QuantizationParams(BaseParams):
    def __init__(self):
        super(QuantizationParams, self).__init__()
        self.add_argument(
            "--quant_w",
            action="store_true",
            help="enable quantization of weights"
        ) 
        self.add_argument(
            "--quant_act",
            action="store_true",
            help="enable quantization of activations"
        )
        self.add_argument(
            "--fuse",
            action="store_true",
            help="fuse batch norm and ReLU cap into weight and bias to simulate chip operation during training"
        )

class TrainParser(QuantizationParams):
    def __init__(self):
        super(TrainParser, self).__init__()
        self.add_argument(
            "--use_gpu",
            action="store_false",
            help="use GPU by default"
        )
        # Training 
        self.add_argument(
            "--train_data_dir",
            type=str,
            default="data/train",
            help="directory where training images are stored",
        )
        self.add_argument(
            "--train_batch_size",
            type=int,
            default=32,
            help="number of images per training batch"
        )
        self.add_argument(
            "--learning_rate",
            type=float,
            default=1e-3,
            help="learning rate"
        )
        self.add_argument(
            "--weight_decay",
            type=float,
            default=1e-4,
            help="weight decay for L2 loss"
        )
        self.add_argument(
            "--num_epochs",
            type=int,
            default=100,
            help="number of epochs to train"
        )
        self.add_argument(
            "--disable_load_opt",
            action="store_true",
            help="by default loads optimizer state; this flag disables that"
        )
        
        # ReLU calibration
        self.add_argument(
            "--percentile",
            type=float,
            default=99.99,
            help="percentile to sample ReLU outputs at"
        )
        self.add_argument(
            "--cal_relu_batches",
            type=int,
            default=10,
            help="how many batches to sample ReLU outputs for"
        )

        # Validation
        self.add_argument(
            "--val_data_dir",
            type=str,
            default="data/val",
            help="directory where validation images are stored",
        )
        self.add_argument(
            "--val_batch_size",
            type=int,
            default=10,
            help="number of images per validation batch"
        )

        # Checkpoints & weights
        self.add_argument(
            "--best_checkpoint_dir",
            type=str,
            default="checkpoints/best",
            help="directory to save best checkpoint"
        )
        self.add_argument(
            "--last_checkpoint_dir",
            type=str,
            default="checkpoints/last",
            help="directory to save latest N checkpoint(s)"
        )
        self.add_argument(
            "--resume_from",
            type=str,
            default=None,
            help="checkpoint to resume training from"
        )

class EvalParser(QuantizationParams):
    def __init__(self):
        super(EvalParser, self).__init__()
        self.add_argument(
            "--use_gpu",
            action="store_false",
            help="use GPU by default"
        )
        self.add_argument(
            "--data_dir",
            type=str,
            default="data/val",
            help="dataset directory",
        )
        self.add_argument(
            "--batch_size",
            type=int,
            default=100,
            help="number of images per batch"
        )

class ConversionParser(BaseParams):
    def __init__(self):
        super(ConversionParser, self).__init__()
        self.add_argument(
            "--ten_bit_act",
            action="store_true",
            help="whether checkpoint has been trained with 10-bit activation for last chip layer"
        )
        self.add_argument(
            "--net_dir",
            type=str,
            default="nets",
            help="location of nets directory"
        )
        self.add_argument(
            "--evaluate_path",
            type=str,
            default=None,
            help="evaluate path for image(s)"
        )
        self.add_argument(
            "--debug",
            type=bool,
            default=False,
            help="debug mode to keep intermediate files"
        )

class ChipInferParser(BaseParams):
    def __init__(self):
        super(ChipInferParser, self).__init__()
        self.add_argument(
            "--ten_bit_act",
            action="store_true",
            help="whether checkpoint has been trained with 10-bit activation for last chip layer"
        )
        self.add_argument(
            "--data_dir",
            type=str,
            default="data/val",
            help="dataset directory for evaluation on chip"
        )
        self.add_argument(
            "--chip_model",
            type=str,
            default="nets/2801_vgg16.model",
            help="dataset directory for evaluation on chip"
        )
        self.add_argument(
            "--image_path",
            type=str,
            help="image path for inference on chip"
        )

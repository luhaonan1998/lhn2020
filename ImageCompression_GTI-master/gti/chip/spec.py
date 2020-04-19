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
from enum import Enum

# supported input image sizes, height == width
DEFAULT_IMAGE_SIZE = 224
ALLOWED_IMAGE_SIZES = {224, 448, 320, 640}

# nets available to each chip (as well as mask bits)
spec_2801 = {
    'vgg16': [3, 3, 3, 1, 1],
    'gnetfc': [3, 3, 1, 1, 1, 1]
} 
spec_2803 = {
    'vgg16': [2]*5,
    'resnet18': [2] * 10,
    'mobilenet': [12]*14,
    'deconv': [2]*6,
    'main_encoder': [2]*3,
    'hyper_encoder': [2]*2,
    'main_decoder': [2]*3,
    'hyper_decoder': [2]*3
}
spec_5801 = {
    'vgg16': [3, 3, 3, 1, 1],
    'mobilenet': [8] * 14,
    'resnet18': [1] * 10
}
specs = {
    '2801':spec_2801,
    '2803':spec_2803,
    '5801':spec_5801
}

# bit schemes for each chip
# mask bits: (weight bits, bias bits)
scheme_2801 = {
    1: (12, 12), 
    3: (12, 18)
}
scheme_2803 = {
    2: (12, 12), 
    5: (12, 16), 
    12: (12,20)
}
scheme_5801 = {
    1: (8, 20), 
    3: (8, 20), 
    8: (8, 20)
}
schemes = {
    '5801':scheme_5801, 
    '2801':scheme_2801, 
    '2803':scheme_2803
}

# CONSTANTS
MIN_SHIFT = 0
MAX_SHIFT = 12 #all chips support up to 15, but
#overflow not accurately accounted for at higher shifts
MAX_ACTIVATION_VALUE = {5: 31.0, 10: 1023.0}

class UpSamplingFillMode(Enum):
    REPEAT = 1
    ZERO = 2

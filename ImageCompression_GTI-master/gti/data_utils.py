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

import torch
from torchvision import transforms, datasets

#[0-1] float -> [0-31] int
class FloatTo5Bit:
    def __call__(self, x):
        out = (((x * 255).int() >> 2) + 1) >> 1
        return torch.clamp(out.float(), 0, 31)

def load_data(train_data_dir, val_data_dir, train_batch_size, val_batch_size):
    
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        FloatToBit()
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)), 
        transforms.ToTensor(),
        FloatTo5Bit()
    ])

    train_dataset = datasets.ImageFolder(
        train_data_dir, 
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        val_data_dir,
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False
    )

    return train_loader, val_loader

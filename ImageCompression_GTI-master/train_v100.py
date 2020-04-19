import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
from utils import torch_msssim
from modules import model
# from nvidia.dali.plugin.pytorch import DALIGenericIterator

from gti.param_parser import TrainParser 

#[0-1] float -> [0-31] int
class FloatTo5Bit:
    def __call__(self, x):
        # out = (((x * 255).int() >> 2) + 1) >> 1
        # out = (x * 255).int()
        # return torch.clamp(out.float(), 0, 31)
        return x

def load_data(train_data_dir, train_batch_size):
    
    train_transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.RandomResizedCrop(size=112),
        transforms.ToTensor(),
        FloatTo5Bit()
    ])

    train_dataset = datasets.ImageFolder(
        train_data_dir, 
        transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=16,
        drop_last=True,
        pin_memory=True
    )
    return train_loader

def train(args):

    TRAINING = True
    CONTEXT = False                              # use context model or not
    METRIC = "PSNR"                              # use metric PSNR or MSSSIM
    print("====> using metric", METRIC)
    lamb = 600.
    
    LOAD_AE = False                              # load existing checkpoint or not
    lr = 1e-4                                    # lr decay from 1e-4 -> 5e-5 -> 1e-5
    
    # model initalization
    image_comp = model.Image_Coder_Context(args).cuda()
    
    if LOAD_AE: 
        image_comp = torch.load('./checkpoints/ae.pt')
    image_comp.train()   
    image_comp = nn.DataParallel(image_comp, device_ids=[0,1]) 
    # Adam optimizer
    optimizer = torch.optim.Adam(image_comp.parameters(),lr=lr)
    
    # MSE loss when metric is "PSNR"
    if METRIC == "MSSSIM":
        loss_func = torch_msssim.MS_SSIM(max_val=1).cuda()
    elif METRIC == "PSNR":
        loss_func = torch.nn.MSELoss()
    
    # training loop
    for epoch in range(25):
        rec_loss, bpp = 0., 0.
    
        train_loader = load_data('/datasets/', args.train_batch_size)
        for step, (batch_x, targets) in enumerate(train_loader):
            # network forward
            batch_x = batch_x.to('cuda')
            num_pixels = batch_x.size()[0]*batch_x.size()[2]*batch_x.size()[3]
            # print(batch_x.size()[2])
            rec, y_main_q, y_hyper, p_main, p_hyper = image_comp(batch_x, TRAINING, CONTEXT)
            
            # distortion between batch_x & rec
            if METRIC == "MSSSIM":
                dloss = 1. - loss_func(rec, batch_x)
            elif METRIC == "PSNR":
                dloss = loss_func(rec, batch_x)
    
            # rate of hyper and main
            train_bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
            train_bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
    
            # loss = lamb * d + rate
            loss = lamb * dloss + train_bpp_main + train_bpp_hyper
    
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            if METRIC=="PSNR":
                torch.nn.utils.clip_grad_norm_(image_comp.parameters(), 10)
            optimizer.step()
            
            if METRIC == "MSSSIM":
                rec_loss = rec_loss + (1. - dloss.item())
                d = 1. - dloss.item()
            elif METRIC == "PSNR":
                rec_loss = rec_loss + dloss.item()
                d = dloss.item()
    
            bpp = bpp+train_bpp_main.item()+train_bpp_hyper.item()
    
            print('epoch', epoch, 'step:', step, '%s:'%(METRIC), d, 'main_bpp:', train_bpp_main.item(), 'hyper_bpp:', train_bpp_hyper.item(), 'lamb', lamb)
    
            cnt = 1000
            if (step+1) % cnt == 0: 
                torch.save(
                   image_comp.module, 
                   os.path.join(
                       "./checkpoints/",
                       'ae_%d_%d_%.8f_%.8f_%0.8f.pt' \
                            % (epoch, step, rec_loss/cnt, bpp/cnt, lamb * rec_loss/cnt + bpp/cnt)
                   )
                )
                rec_loss, bpp = 0., 0.

if __name__ == "__main__":
    args = TrainParser().parse_args()
    train(args)

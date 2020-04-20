import os
import torch
import torch.nn as nn
import numpy as np
from glob import glob
from PIL import Image
from utils import torch_msssim
from modules import model

from gti.param_parser import TrainParser 

def test(args, im_dir, CONTEXT=False, crop=None):

    TRAINING = False
    GPU = True
    # read image
    precise = 16
    # print('====> Encoding Image:', im_dir)
    
    img = Image.open(im_dir)
    img = np.array(img)/255.0
    H, W, _ = img.shape

    num_pixels = H * W
    
    C = 3
    if crop == None:
        tile = 16.
    else:
        tile = crop * 1.0

    H_PAD = int(tile * np.ceil(H / tile))
    W_PAD = int(tile * np.ceil(W / tile))
    im = np.zeros([H_PAD, W_PAD, 3], dtype='float32')
    im[:H, :W, :] = img[:,:,:3]
    im = torch.FloatTensor(im)

    # model initalization
    image_comp = model.Image_Coder_Context(args)
    image_comp = torch.load('/checkpoints/ae.pt')
    image_comp.eval() 
    if CONTEXT:
        context = torch.load('checkpoints/best/context.pt')   

    if GPU:
        image_comp.cuda()
        #msssim_func = msssim_func.cuda()
        im = im.cuda()

    im = im.permute(2, 0, 1).contiguous()
    im = im.view(1, C, H_PAD, W_PAD)

    mssim_func = torch_msssim.MS_SSIM(max_val=1).cuda()
    if crop == None:
        output, y_main_q, y_hyper, p_main, p_hyper = image_comp(im, TRAINING, CONTEXT)
        if CONTEXT:
            p_main, _ = context(y_main_q, p_main)
        bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
        bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
    else:
        bpp_main, bpp_hyper = 0., 0.
        output = torch.zeros(1,C,H_PAD,W_PAD).cuda()
        for i in range(int(H_PAD/crop)):
            for j in range(int(W_PAD/crop)):
                rec_tile, _, _, p_main, p_hyper = image_comp(im[:,:,i*int(crop):(i+1)*int(crop),j*int(crop):(j+1)*int(crop)], TRAINING, CONTEXT)
                output[:,:,i*int(crop):(i+1)*int(crop),j*int(crop):(j+1)*int(crop)] = rec_tile
                bpp_hyper += torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
                bpp_main += torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
                print(i,j,bpp_hyper,bpp_main)
    # rate of hyper and main
    # bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
    # bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
    bpp = bpp_main + bpp_hyper

    output_ = torch.clamp(output, min=0., max=1.0)
    out = output_.data[0].cpu().numpy()
    out = np.round(out * 255.0) 
    out = out.astype('uint8')
    
    #ms-ssim
    mssim = mssim_func(im[:,:,:H,:W].cuda(),output_[:,:,:H,:W].cuda())
    
    #psnr float
    mse =  torch.mean((im[:,:,:H,:W] - output_[:,:,:H,:W]) * (im[:,:,:H,:W] - output_[:,:,:H,:W]))
    psnr = 10. * np.log(1.0/mse.item())/ np.log(10.)
    
    print(im_dir, "bpp(main/hyper):%f (%f / %f)"%(bpp, bpp_main, bpp_hyper), "PSNR:", psnr)
    out = out.transpose(1, 2, 0)
    img = Image.fromarray(out[:H, :W, :])
    img.save("rec.png")
    #psnr uint8
    #mse_i =  torch.mean((im - torch.Tensor([out/255.0]).cuda()) * (im - torch.Tensor([out/255.0]).cuda()))
    #psnr_i = 10. * np.log(1.0/mse_i.item())/ np.log(10.)
    # print("bpp: %f PSNR: %f")
    return bpp, psnr

if __name__ == "__main__":
    args = TrainParser().parse_args()
    dirs = glob('./kodak/kodim*.png')
    bpps, psnrs = 0., 0.
    with torch.no_grad():
        for dir in dirs:
            bpp,psnr = test(args, dir, CONTEXT=False, crop=None)
            bpps += bpp
            psnrs += psnr
    print("bpps:%0.4f, psnr:%0.4f"%(bpps/24.0, psnrs/24.0))

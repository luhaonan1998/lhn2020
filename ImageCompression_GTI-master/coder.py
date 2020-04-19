import os, time
import torch
import torch.nn as nn
import numpy as np
from glob import glob
from PIL import Image
from utils import torch_msssim,ops
from modules import model
from modules.context import Context
from gti.param_parser import TrainParser 

import AE

TRAINING = False
GPU = True
precise = 16

def encode(args, im_dir, out_dir, CONTEXT=False, crop=None):
    ################ read image #########################
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
    print('====> Encoding Image:', im_dir, "%dx%d"%(H, W))

    ############### model initalization ################
    image_comp = model.Image_Coder_Context(args)
    image_comp = torch.load('./checkpoints/ae_24_999_0.00130410_0.36951993_1.15198196.pt')
    # image_comp = torch.load('checkpoints/best/ae.pt')
    image_comp.eval() 
    if CONTEXT:
        context = torch.load('./checkpoints/context.pt')   

    mssim_func = torch_msssim.MS_SSIM(max_val=1).cuda()
    if GPU:
        image_comp.cuda()
        #msssim_func = msssim_func.cuda()
        im = im.cuda()

    im = im.permute(2, 0, 1).contiguous()
    im = im.view(1, C, H_PAD, W_PAD)

    if crop == None:
        y_main, y_hyper = image_comp.encoder(im)
        y_main_q = torch.round(y_main)
        y_hyper_q = torch.round(y_hyper)
        
        p_raw = image_comp.hyper_dec(y_hyper_q) # [1,2*C,H,W]
        mean = p_raw[:,:256,:,:]
        scale = p_raw[:,256:,:,:]
        scale = ops.Low_bound.apply(torch.abs(scale), 1e-6)
        Datas = torch.reshape(y_main_q, [-1]).cpu().numpy().astype(np.int).tolist()
        Means = torch.reshape(mean, [-1]).cpu().numpy().tolist()
        Scales = torch.reshape(scale, [-1]).cpu().numpy().tolist()
        
        ## Main Arith Encode
        Min_V = min(Datas)
        Max_V = max(Datas)
        AE.encode(Datas, Means, Scales, out_dir)

        ## Hyper Arith 
        Min_V_HYPER = torch.min(y_hyper_q).cpu().numpy().astype(np.int)
        Max_V_HYPER = torch.max(y_hyper_q).cpu().numpy().astype(np.int)
        Datas_hyper = torch.reshape(y_hyper_q, [256, -1]).cpu().numpy().astype(np.int).tolist()
        
        print(Min_V_HYPER, Max_V_HYPER)
        sample = np.arange(Min_V_HYPER, Max_V_HYPER+1+1) # [Min_V - 0.5 , Max_V + 0.5]
        sample = np.tile(sample, [256,1,1])
        lower = torch.sigmoid(image_comp.factorized_entropy_func._logits_cumulative(torch.FloatTensor(sample).cuda() - 0.5, stop_gradient=False)) 
        cdf_h = lower.data.cpu().numpy()*((1 << precise)- (Max_V_HYPER - Min_V_HYPER + 1)) # [N1, 1, Max - Min]
        cdf_h = cdf_h.astype(np.int) + sample.astype(np.int) - Min_V_HYPER
        cdf_hyper = np.reshape(np.tile(cdf_h, [len(Datas_hyper[0]), 1, 1, 1]),[len(Datas_hyper[0]), 256, -1])
        
        # Datas_hyper [256, N], cdf_hyper [256,1,X]
        Cdf_0, Cdf_1 = [],[]
        print(len(Datas_hyper[0]))
        # print(cdf_hyper.shape)
        for i in range(256):
            # print(max(Datas_hyper[i]))
            Cdf_0.extend(list(map(lambda x, y: int(y[x - Min_V_HYPER]), Datas_hyper[i], cdf_hyper[:,i,:])))   # Cdf[Datas - Min_V]
            Cdf_1.extend(list(map(lambda x, y: int(y[x - Min_V_HYPER]), Datas_hyper[i], cdf_hyper[:,i,1:]))) # Cdf[Datas + 1 - Min_V]
        print(len(Cdf_0))
        AE.encode_cdf(Cdf_0, Cdf_1, "./compressed.bin")

        AE.init_decoder("./compressed.bin", Min_V_HYPER, Max_V_HYPER)
        T2 = time.time()
        Recons = []
        for i in range(256):
            for j in range(96):
                # print(cdf_h[i,0,:])
                Recons.append(AE.decode_cdf(cdf_h[i,0,:].tolist()))
        TE2 = time.time() - T2

        error_number = 0
        for i in range(256):
            for j in range(96):
                if Datas_hyper[i][j] != Recons[96*i+j]:
                    error_number += 1
                    print(Datas_hyper[i][j], Recons[96*i+j])
        print("Error Number: ", error_number)

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
    dirs = glob('/kodak/kodim*.png')
    bpps, psnrs = 0., 0.
    with torch.no_grad():
        for dir in dirs:
            out_dir = dir.replace('.png' , '.dic')
            bpp,psnr = encode(args, dir, out_dir, CONTEXT=False, crop=None)
            bpps += bpp
            psnrs += psnr
    print("bpps:%0.4f, psnr:%0.4f"%(bpps/24.0, psnrs/24.0))

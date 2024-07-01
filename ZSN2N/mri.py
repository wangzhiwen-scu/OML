# https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b#scrollTo=G0plKGddguir
# https://openaccess.thecvf.com/content/CVPR2023/papers/Mansour_Zero-Shot_Noise2Noise_Efficient_Image_Denoising_Without_Any_Data_CVPR_2023_paper.pdf


import numpy as np 
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# import requests
# from io import BytesIO

import scipy.io as sio
from numpy.fft import fftshift
import sys
sys.path.append('.') # 
from layers.mask_layer import Hand_Tailed_Mask_Layer4ZSN2N as Hand_Tailed_Mask_Layer, Hand_Tailed_Mask_Layer4SSL, ComplexAbs
from model_backbones.dualdomain import DualDoRecNet4SSL as DualDoRecNet, fft, ifft

#Enter device here, 'cuda' for GPU, and 'cpu' for CPU
device = 'cuda'


path = "./ZSN2N/data/00023"
clean_img = torch.load(path).unsqueeze(0) 


clean_img = torch.mean(clean_img, dim=1, keepdim=True)
clean_img = F.interpolate(clean_img, size=(256, 256), mode='bilinear', align_corners=False)
clean_img = clean_img.to(device, dtype=torch.float)
print(clean_img.shape) #B C H W

noise_type = 'gauss' # Either 'gauss' or 'poiss'
noise_level = 25     # Pixel range is 0-255 for Gaussian, and 0-1 for Poission

# def add_noise(x,noise_level):
#     if noise_type == 'gauss':
#         noisy = x + torch.normal(0, noise_level/255, x.shape)
#         noisy = torch.clamp(noisy,0,1)
        
#     elif noise_type == 'poiss':
#         noisy = torch.poisson(noise_level * x)/noise_level
#     return noisy

# noisy_img = add_noise(clean_img, noise_level)
sample_net = Hand_Tailed_Mask_Layer(0.33, 'cartesian', (256, 256), resize_size=[256,256]).to(device, dtype=torch.float)
# subsample_net1 = Hand_Tailed_Mask_Layer4SSL(0.33, 'cartesian', (240, 240), resize_size=[240,240]).to(device, dtype=torch.float)
# subsample_net2 = Hand_Tailed_Mask_Layer4SSL(0.33, 'cartesian', (240, 240), resize_size=[240,240]).to(device, dtype=torch.float)
# clean_img = clean_img.to(device)
fft_func = fft
ifft_func = ifft
abs_func = ComplexAbs()
out = sample_net(clean_img)
u_k, u_img_abs, u_img, batch_mask = out['masked_k'], out['complex_abs'], out['zero_filling'],  out['batch_trajectories']


class network(nn.Module):
    def __init__(self,n_chan,chan_embed=48):
        super(network, self).__init__()
        
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan,chan_embed,3,padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x



# n_chan = clean_img.shape[1]
n_chan=2
# model = network(n_chan)
model = DualDoRecNet()
model = model.to(device, dtype=torch.float)
print("The number of parameters of the network is: ",  sum(p.numel() for p in model.parameters() if p.requires_grad))

# def pair_downsampler(img):
#     #img has shape B C H W
#     c = img.shape[1]
    
#     filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
#     filter1 = filter1.repeat(c,1, 1, 1)
    
#     filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
#     filter2 = filter2.repeat(c,1, 1, 1)
    
#     output1 = F.conv2d(img, filter1, stride=2, groups=c)
#     output2 = F.conv2d(img, filter2, stride=2, groups=c)

#     return output1, output2

torch.manual_seed(0)
# dropout_mask_1 = torch.rand(1, 1, 240, 240).to(device, dtype=torch.float)
# dropout_mask_1 = (dropout_mask_1 > 0.5).to(device, dtype=torch.float)
# dropout_mask_2 = 1 - dropout_mask_1
path1 = "ZSN2N/data/ssl_mask/selecting_mask/mask_2.00x_acs16.mat"
path2 = "ZSN2N/data/ssl_mask/selecting_mask/mask_2.50x_acs16.mat"
dropout_mask_1 = sio.loadmat(path1)['mask']
dropout_mask_2 = sio.loadmat(path2)['mask']
dropout_mask_1 = fftshift(dropout_mask_1, axes=(-2, -1))
dropout_mask_2 = fftshift(dropout_mask_2, axes=(-2, -1))
dropout_mask_1 = torch.tensor(dropout_mask_1).to(device, dtype=torch.float)
dropout_mask_2 = torch.tensor(dropout_mask_2).to(device, dtype=torch.float)

mask = batch_mask
mask1 = batch_mask * dropout_mask_1
mask2 = batch_mask * dropout_mask_2


def pair_downsampler_kspace(u_k, mask1, mask2):

    uu_k1 = mask1 * u_k
    uu_k2 = mask2 * u_k
    
    return uu_k1, uu_k2

uu_k1, uu_k2 = pair_downsampler_kspace(u_k, mask1, mask2)

img1 = abs_func(ifft_func(uu_k1))
img2 = abs_func(ifft_func(uu_k2))

img0 = u_img_abs.cpu().squeeze(0).permute(1,2,0)
img1 = img1.cpu().squeeze(0).permute(1,2,0)
img2 = img2.cpu().squeeze(0).permute(1,2,0)

fig, ax = plt.subplots(1, 3,figsize=(15, 15))

ax[0].imshow(img0)
ax[0].set_title('Noisy Img')

ax[1].imshow(img1)
ax[1].set_title('First downsampled')

ax[2].imshow(img2)
ax[2].set_title('Second downsampled')


# Save the figure with all three images
plt.tight_layout()
plt.savefig('./ZSN2N/results/subsampled_mri.png', bbox_inches='tight')

# # Alternatively, you can save each image separately:
# plt.figure(1)
# plt.imshow(img0)
# plt.title('Noisy Img')
# plt.savefig('img0.png')

# plt.figure(2)
# plt.imshow(img1)
# plt.title('First downsampled')
# plt.savefig('img1.png')

# plt.figure(3)
# plt.imshow(img2)
# plt.title('Second downsampled')
# plt.savefig('img2.png')

def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt,pred)

def loss_func(u_k, mask, mask1, mask2):

    uu_k1, uu_k2 = pair_downsampler_kspace(u_k, mask1, mask2)

    uu_img1 = fft_func(uu_k1)
    uu_img2 = fft_func(uu_k2)

    pred1 =  model(uu_k1, mask1)
    pred2 =  model(uu_k2, mask2)

    loss_res = 1/2*(mse(uu_img1, pred2)+mse(uu_img2, pred1))
    
    u_img = fft_func(u_k)
    pred =  model(u_k, mask)

    pred_k = fft_func(pred)

    rec_k1, rec_k2 = pair_downsampler_kspace(pred_k, mask1, mask2)

    # pred1_k = ifft_func(pred1)
    # pred2_k = ifft_func(pred2)

    rec_img1 = fft_func(rec_k1)
    rec_img2 = fft_func(rec_k2)

    loss_cons=1/2*(mse(pred1, rec_img1) + mse(pred2, rec_img2))
    
    loss = loss_res + loss_cons

    return loss

def loss_func_official(noisy_img):
    noisy1, noisy2 = pair_downsampler_kspace(noisy_img)

    pred1 =  noisy1 - model(noisy1)
    pred2 =  noisy2 - model(noisy2)
    
    loss_res = 1/2*(mse(noisy1,pred2)+mse(noisy2,pred1))
    
    noisy_denoised =  noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler_kspace(noisy_denoised)
    
    loss_cons=1/2*(mse(pred1,denoised1) + mse(pred2,denoised2))
    
    loss = loss_res + loss_cons

    return loss

def loss_func_ssl_v2(u_k, mask, mask1, mask2):
    uu_k1, uu_k2 = pair_downsampler_kspace(u_k, mask1, mask2)

    uu_img1 = fft_func(uu_k1)
    uu_img2 = fft_func(uu_k2)
    u_img = fft_func(u_k)

    full_mask = torch.ones_like(mask1)
    pred1 = uu_img1 - model(uu_k1, mask1)
    pred2 = uu_img2 - model(uu_k2, mask2)
    loss_res = 1/2*(mse(uu_img1,pred2)+mse(uu_img2,pred1))

    # noisy_denoised =  u_img - model(u_k, mask)
    noisy_denoised = model(u_k, mask)
    rec_k = fft_func(noisy_denoised)

    denoised1, denoised2 = pair_downsampler_kspace(rec_k, mask1, mask2)
    denoised1_img = fft_func(denoised1)
    denoised2_img = fft_func(denoised2)
    loss_cons=1/2*(mse(pred1,denoised1_img) + mse(pred2,denoised2_img))

    comp_mask1 = full_mask - mask1
    comp_mask2 = full_mask - mask2
    comp_img1 = ifft_func(fft_func(pred1) * comp_mask1)
    comp_img2 = ifft_func(fft_func(pred2) * comp_mask2)
    loss = mse(comp_img1, comp_img2) + loss_res + loss_cons


    return loss

def loss_func_ssl(u_k, mask, mask1, mask2):
    uu_k1, uu_k2 = pair_downsampler_kspace(u_k, mask1, mask2)
    uu_img1 = fft_func(uu_k1)
    uu_img2 = fft_func(uu_k2)
    full_mask = torch.ones_like(mask1)
    pred1 =  model(uu_k1, mask1)
    pred2 =  model(uu_k2, mask2)
    comp_mask1 = full_mask - mask1
    comp_mask2 = full_mask - mask2
    comp_img1 = ifft_func(fft_func(pred1) * comp_mask1)
    comp_img2 = ifft_func(fft_func(pred2) * comp_mask2)
    loss = mse(comp_img1, comp_img2)
    return loss



def train(model, optimizer, u_k, mask, mask1, mask2):
  
    # loss = loss_func(u_k, mask, mask1, mask2)
    # loss = loss_func_ssl(u_k, mask, mask1, mask2)
    loss = loss_func_ssl_v2(u_k, mask, mask1, mask2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, u_k, clean_img, mask):
    
    with torch.no_grad():
        # u_img = ifft_func(u_k)
        pred =model(u_k, mask)

        # pred = abs_func(pred)
        pred = torch.norm(pred, keepdim=True, dim=1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10*np.log10(1/MSE)
    
    return PSNR

def denoise(model, u_k, mask):
    u_img = ifft_func(u_k)
    with torch.no_grad():
        # pred = torch.clamp( u_img - model(u_k, mask),0,1)
        pred = model(u_k, mask)
        pred = torch.norm(pred, keepdim=True, dim=1)
    return pred

# max_epoch = 2000     # training epochs
max_epoch = 200     # training epochs

lr = 0.001           # learning rate
step_size = 150     # number of epochs at which learning rate decays
gamma = 0.5          # factor by which learning rate decays

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

for epoch in tqdm(range(max_epoch)):
    train(model, optimizer, u_k, mask, mask1, mask2)
    scheduler.step()

PSNR = test(model, u_k, clean_img, mask)
print(PSNR)

denoised_img = denoise(model, u_k, mask)

denoised = denoised_img.cpu().squeeze(0).squeeze(0)
clean = clean_img.cpu().squeeze(0).squeeze(0)
noisy = img0.cpu().squeeze(0).squeeze(0)

fig, ax = plt.subplots(1, 3,figsize=(15, 15))

ax[0].imshow(clean)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Ground Truth')

ax[1].imshow(noisy)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Noisy Img')
noisy_psnr = 10*np.log10(1/mse(noisy, clean).item())
ax[1].set(xlabel= str(round(noisy_psnr,2)) + ' dB')

ax[2].imshow(denoised)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title('Denoised Img')
ax[2].set(xlabel= str(round(PSNR,2)) + ' dB')

plt.tight_layout()
plt.savefig('./ZSN2N/results/reconstruced_mri.png', bbox_inches='tight')
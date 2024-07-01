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

# import requests
# from io import BytesIO
import scipy.io as sio
#Enter device here, 'cuda' for GPU, and 'cpu' for CPU
device = 'cuda'


# path = "./ZSN2N/data/00023"
# clean_img = torch.load(path).unsqueeze(0) 
# print(clean_img.shape) #B C H W

path_ldct = "ZSN2N/data/mayo/data_0113_img.mat"
path_gt = "ZSN2N/data/mayo/data_0113_gt.mat"

noisy_img = sio.loadmat(path_ldct)['data']
clean_img = sio.loadmat(path_gt)['data'] / 3.84

noisy_img = torch.tensor(np.clip(noisy_img, 0.42, 0.62)).view(1, 1, 256, 256).cuda()
clean_img = torch.tensor(np.clip(clean_img, 0.42, 0.62)).view(1, 1, 256, 256).cuda()
# clean_img = torch.mean(clean_img, dim=1, keepdim=True)
# clean_img = F.interpolate(clean_img, size=(240, 240), mode='bilinear', align_corners=False)
print(clean_img.shape) #B C H W

# noise_type = 'gauss' # Either 'gauss' or 'poiss'
# noise_level = 25     # Pixel range is 0-255 for Gaussian, and 0-1 for Poission

# def add_noise(x,noise_level):
    
#     if noise_type == 'gauss':
#         noisy = x + torch.normal(0, noise_level/255, x.shape)
#         noisy = torch.clamp(noisy,0,1)
        
#     elif noise_type == 'poiss':
#         noisy = torch.poisson(noise_level * x)/noise_level
    
#     return noisy

# noisy_img = add_noise(clean_img, noise_level)    

clean_img = clean_img.to(device)
noisy_img = noisy_img.to(device)

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

n_chan = clean_img.shape[1]
model = network(n_chan)
model = model.to(device)
print("The number of parameters of the network is: ",  sum(p.numel() for p in model.parameters() if p.requires_grad))

def pair_downsampler(img):
    #img has shape B C H W
    c = img.shape[1]
    
    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c,1, 1, 1)
    
    filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c,1, 1, 1)
    
    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2

img1, img2 = pair_downsampler(noisy_img)

img0 = noisy_img.cpu().squeeze(0).permute(1,2,0)
img1 = img1.cpu().squeeze(0).permute(1,2,0)
img2 = img2.cpu().squeeze(0).permute(1,2,0)

fig, ax = plt.subplots(1, 3,figsize=(15, 15))



ax[0].imshow(img0, cmap='gray')
ax[0].set_title('Noisy Img')

ax[1].imshow(img1, cmap='gray')
ax[1].set_title('First downsampled')

ax[2].imshow(img2, cmap='gray')
ax[2].set_title('Second downsampled')


# Save the figure with all three images
plt.tight_layout()
plt.savefig('./ZSN2N/results/ct.png', dpi=300, bbox_inches='tight')

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

def loss_func(noisy_img):
    noisy1, noisy2 = pair_downsampler(noisy_img)

    pred1 =  noisy1 - model(noisy1)
    pred2 =  noisy2 - model(noisy2)
    
    loss_res = 1/2*(mse(noisy1,pred2)+mse(noisy2,pred1))
    
    noisy_denoised =  noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    
    loss_cons=1/2*(mse(pred1,denoised1) + mse(pred2,denoised2))
    
    loss = loss_res + loss_cons

    return loss

def train(model, optimizer, noisy_img):
  
  loss = loss_func(noisy_img)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return loss.item()

def test(model, noisy_img, clean_img):
    
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img),0,1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10*np.log10(1/MSE)
    
    return PSNR

def denoise(model, noisy_img):
    
    with torch.no_grad():
        pred = torch.clamp( noisy_img - model(noisy_img),0,1)
    
    return pred 


max_epoch = 2000     # training epochs
lr = 0.001           # learning rate
step_size = 1500     # number of epochs at which learning rate decays
gamma = 0.5          # factor by which learning rate decays

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

for epoch in tqdm(range(max_epoch)):
    train(model, optimizer, noisy_img)
    scheduler.step()

PSNR = test(model, noisy_img, clean_img)
print(PSNR)

denoised_img = denoise(model, noisy_img)

denoised = denoised_img.cpu().squeeze(0).permute(1,2,0)
clean = clean_img.cpu().squeeze(0).permute(1,2,0)
noisy = noisy_img.cpu().squeeze(0).permute(1,2,0)

fig, ax = plt.subplots(1, 3,figsize=(15, 15))

ax[0].imshow(clean, cmap='gray')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('Ground Truth')

ax[1].imshow(noisy, cmap='gray')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('Noisy Img')
noisy_psnr = 10*np.log10(1/mse(noisy_img,clean_img).item())
ax[1].set(xlabel= str(round(noisy_psnr,2)) + ' dB')

ax[2].imshow(denoised, cmap='gray')
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title('Denoised Img')
ax[2].set(xlabel= str(round(PSNR,2)) + ' dB')

plt.tight_layout()
plt.savefig('./ZSN2N/results/ct_denoised.png', dpi=600, bbox_inches='tight')
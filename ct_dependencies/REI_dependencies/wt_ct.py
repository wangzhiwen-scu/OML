import torch
import torch.nn as nn
from torch.autograd import Function
import ctlib
from json import load
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
# import h5py
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import transforms
from scipy.io import loadmat,savemat
import torch.utils.data as Data
from PIL import Image
# import torchvision.transforms as T
import torch.utils.data 
import matplotlib.pyplot as plt 

from .transform import Rotate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




options = torch.tensor([160, 640, 416, 416, 0.006641, 0.012858, 0, 0.009817477*4,
                5.95, 4.906, 0, 0]).cuda()

class prj_fun(Function):
    @staticmethod
    def forward(self, image, options):
        self.save_for_backward(options)
        return ctlib.projection(image, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        grad_input = ctlib.projection_t(grad_output.contiguous(), options)
        return grad_input, None

class prj_t_fun(Function):
    @staticmethod
    def forward(self, proj, options):
        self.save_for_backward(options)
        return ctlib.projection_t(proj, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        grad_input = ctlib.projection(grad_output.contiguous(), options)
        return grad_input, None

class projector(nn.Module):
    def __init__(self, options):
        super(projector, self).__init__()
        self.options = options
        
    def forward(self, image):
        return prj_fun.apply(image, self.options)

class projector_t(nn.Module):
    def __init__(self):
        super(projector_t, self).__init__()
        
    def forward(self, proj, options):
        return prj_t_fun.apply(proj, options)
class bprj_fun(Function):
    @staticmethod
    def forward(self, proj, options):
        self.save_for_backward(options)
        return ctlib.backprojection(proj, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        grad_input = ctlib.backprojection_t(grad_output.contiguous(), options)
        return grad_input, None

class backprojector(nn.Module):
    def __init__(self):
        super(backprojector, self).__init__()
        
    def forward(self, image, options):
        return bprj_fun.apply(image, options)

class FBP(nn.Module):
    def __init__(self, options):
        super(FBP, self).__init__()
        dets = int(options[1])
        dDet = options[5]
        s2r = options[8]
        d2r = options[9]
        virdet = dDet * s2r / (s2r + d2r)
        filter = torch.empty(2 * dets - 1)
        pi = torch.acos(torch.tensor(-1.0))
        for i in range(filter.size(0)):
            x = i - dets + 1
            if abs(x) % 2 == 1:
                filter[i] = -1 / (pi * pi * x * x * virdet * virdet)
            elif x == 0:
                filter[i] = 1 / (4 * virdet * virdet)
            else:
                filter[i] = 0
        filter = filter.view(1,1,1,-1)
        w = torch.arange((-dets / 2 + 0.5) * virdet, dets / 2 * virdet, virdet).cuda()
        w = s2r / torch.sqrt(s2r ** 2 + w ** 2)
        w = w.view(1,1,1,-1) * virdet
        self.w = nn.Parameter(w, requires_grad=False)
        self.filter = nn.Parameter(filter, requires_grad=False)
        self.options = nn.Parameter(options, requires_grad=False)
        self.backprojector = backprojector()
        self.dets = dets
        self.coef = pi / options[0]

    def forward(self, projection):
        p = projection * self.w
        p = torch.nn.functional.conv2d(p, self.filter, padding=(0,self.dets-1))
        recon = self.backprojector(p, self.options)
        recon = recon * self.coef
        return recon
class fidelity_module(nn.Module):
    def __init__(self, options):
        super(fidelity_module, self).__init__()
        hid_channels=48
        kernel_size=5
        padding=2
        self.options = nn.Parameter(options, requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(1).squeeze())
        self.projector = projector(options)
        self.projector_t = projector_t()
        self.sinonet = nn.Sequential(
            nn.Conv2d(1, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, 1, kernel_size=kernel_size, padding=padding)
        )
        
    def forward(self, input_data, proj, mask):
        
        sino_in = self.projector(input_data)
        tempy = self.sinonet(sino_in)*(1-mask)+proj*mask
        temp = sino_in-tempy
        intervening_res = self.projector_t(temp, self.options)
        out = input_data - self.weight * intervening_res
        return out

class CT():
    def __init__(self, img_width, radon_view, uniform=True, circle = False, device='cuda:0'):
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
        else:
            theta = torch.arange(radon_view)
        # options = torch.tensor([radon_view, 640, img_width, img_width, 0.006641, 0.012858, 0, 0.009817477*640/radon_view,
        #         5.95, 4.906, 0, 0]).cuda()
        options = torch.tensor([120, 640, 416, 416, 0.006641, 0.012858, 0, 0.009817477*640/120,
                5.95, 4.906, 0, 0]).cuda()
        
        self.radon = projector(options).cuda()
        self.iradon = FBP(options).cuda()

    def A(self, x):
        return self.radon(x)

    def A_dagger(self, y):
        return self.iradon(y)
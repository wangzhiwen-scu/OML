# https://github.com/xiawj-hub/Physics-Model-Data-Driven-Review/blob/main/recon/models/LEARN.py

import torch
import torch.nn as nn
from torch.autograd import Function
import ctlib

class prj_fun(Function):
    @staticmethod
    def forward(self, image, options):
        self.save_for_backward(options)
        return ctlib.projection(image, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        # grad_input = ctlib.projection_t(grad_output.contiguous(), options)
        grad_input = ctlib.fbp(grad_output.contiguous(), options)
        return grad_input, None

class prj_t_fun(Function):
    @staticmethod
    def forward(self, proj, options):
        self.save_for_backward(options)
        # return ctlib.projection_t(proj, options)
        return ctlib.fbp(proj, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        grad_input = ctlib.projection(grad_output.contiguous(), options)
        return grad_input, None

class projector(nn.Module):
    def __init__(self):
        super(projector, self).__init__()
        
    def forward(self, image, options):
        return prj_fun.apply(image, options)

class projector_t(nn.Module):
    def __init__(self):
        super(projector_t, self).__init__()
        
    def forward(self, proj, options):
        return prj_t_fun.apply(proj, options)

class fidelity_module(nn.Module):
    def __init__(self, options):
        super(fidelity_module, self).__init__()
        self.options = nn.Parameter(options, requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(1).squeeze())
        self.projector = projector()
        self.projector_t = projector_t()
        
    def forward(self, input_data, proj):
        temp = self.projector(input_data, self.options) - proj
        intervening_res = self.projector_t(temp, self.options)
        out = input_data - self.weight * intervening_res
        return out

class Iter_block(nn.Module):
    def __init__(self, hid_channels, kernel_size, padding, options):
        super(Iter_block, self).__init__()
        self.block1 = fidelity_module(options)
        self.block2 = nn.Sequential(
            nn.Conv2d(1, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, 1, kernel_size=kernel_size, padding=padding)
        )
        self.relu = nn.ReLU(inplace=True)      

    def forward(self, input_data, proj):
        tmp1 = self.block1(input_data, proj)
        tmp2 = self.block2(input_data)
        output = tmp1 + tmp2
        output = self.relu(output)
        return output

class LEARN(nn.Module):
    def __init__(self, options, block_num=50, hid_channels=48, kernel_size=5, padding=2):
        super(LEARN, self).__init__()
        self.model = nn.ModuleList([Iter_block(hid_channels, kernel_size, padding, options) for i in range(block_num)])
        for module in self.modules():
            if isinstance(module, fidelity_module):
                module.weight.data.zero_()
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_data, proj):
        x = input_data
        for index, module in enumerate(self.model):
            x = module(x, proj)
        return x
# https://github.com/xiawj-hub/Physics-Model-Data-Driven-Review/blob/main/recon/models/LEARN.py

import torch
import torch.nn as nn
from torch.autograd import Function
import ctlib


from .EI_dependencies.ct import CT

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class fidelity_module(nn.Module):
    def __init__(self, physics):
        super(fidelity_module, self).__init__()
        # self.options = nn.Parameter(options, requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(1).squeeze())
        # self.projector = projector()
        # self.projector_t = projector_t()

        self.projector = physics.A
        self.projector_t = physics.A_dagger
        
    def forward(self, input_data, proj):
        temp = self.projector(input_data) - proj
        intervening_res = self.projector_t(temp)
        out = input_data - self.weight * intervening_res
        return out

class Iter_block(nn.Module):
    def __init__(self, hid_channels, kernel_size, padding, physics):
        super(Iter_block, self).__init__()
        self.block1 = fidelity_module(physics)
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

        optionss = options.detach().cpu().numpy()

        img_hw = int(optionss[2])
        n_views = int(optionss[0])



        self.physics = CT(img_hw, n_views, circle=False, device=device)

        self.model = nn.ModuleList([Iter_block(hid_channels, kernel_size, padding, self.physics) for i in range(block_num)])
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
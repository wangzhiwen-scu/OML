# -*- coding: utf-8 -*-
"""
Created on Nov. 3, 2020
enhanced version of FISTA-Net
(1) with learned gradient matrix
(2) 
@author: XIANG
# https://github.com/jinxixiang/FISTA-Net/blob/master/M5FISTANetPlus.py
# https://github.com/jinxixiang/FISTA-Net/blob/master/solver.py 
# https://arxiv.org/pdf/2008.02683.pdf
"""

import torch 
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np 
import os

import sys
sys.path.append(".")
from model_backbones.ista_net import rfft, fft, ifft, FFT_Mask_ForBack

from model_backbones.lighting_unet import LightDoubleConv, Down, Up, OutConv
# from model_backbones.ms_ssim_loss import MS_SSIM_L1_LOSS
from model_backbones import ssim_loss

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)


# define basic block of FISTA-Net
class  BasicBlock(nn.Module):
    """docstring for  BasicBlock"""

    def __init__(self, features=32):
        super(BasicBlock, self).__init__()
        #self.lambda_step = nn.Parameter(torch.Tensor([0.2]))
        #self.soft_thr = nn.Parameter(torch.Tensor([0.05]))
        self.Sp = nn.Softplus()

        # self.conv_D = nn.Conv2d(1, features, (3,3), stride=1, padding=1)
        self.conv_D = nn.Conv2d(2, features, (3,3), stride=1, padding=1)
        self.conv1_forward = nn.Conv2d(features, features, (3,3), stride=1, padding=1)
        self.conv2_forward = nn.Conv2d(features, features, (3,3), stride=1, padding=1)
        self.conv3_forward = nn.Conv2d(features, features, (3,3), stride=1, padding=1)
        self.conv4_forward = nn.Conv2d(features, features, (3,3), stride=1, padding=1)
        
        self.conv1_backward = nn.Conv2d(features, features, (3,3), stride=1, padding=1)
        self.conv2_backward = nn.Conv2d(features, features, (3,3), stride=1, padding=1)
        self.conv3_backward = nn.Conv2d(features, features, (3,3), stride=1, padding=1)
        self.conv4_backward = nn.Conv2d(features, features, (3,3), stride=1, padding=1)
        # self.conv_G = nn.Conv2d(features, 1, (3,3), stride=1, padding=1)
        self.conv_G = nn.Conv2d(features, 2, (3,3), stride=1, padding=1)


    def forward(self, x, fft_forback, mask, PhiTb, lambda_step, soft_thr):
        """_summary_

        Args:
            x (_type_): is rec_u_img
            PhiTPhi (_type_): _description_
            PhiTb (_type_): _description_
            mask (_type_): _description_
            lambda_step (_type_): _description_
            soft_thr (_type_): _description_

        Returns:
            _type_: _description_
        """
        # convert data format from (batch_size, channel, pnum, pnum) to (circle_num, batch_size)
        # pnum = x.size()[2]
        # x = x.view(x.size()[0], x.size()[1], pnum*pnum, -1)   # (batch_size, channel, pnum*pnum, 1)
        # x = torch.squeeze(x, 1)
        # x = torch.squeeze(x, 2).t()             
        # x = mask.mm(x)
        
        # x = x - self.Sp(lambda_step)  * PhiTPhi.mm(x) + self.Sp(lambda_step) * PhiTb # what is self.Sp=nn.Softplus() ?
        
        # x = x - self.lambda_step * fft_forback(x, mask) + self.lambda_step * PhiTb
        x = x - self.Sp(lambda_step) * fft_forback(x, mask) + self.Sp(lambda_step) * PhiTb
        # convert (circle_num, batch_size) to (batch_size, channel, pnum, pnum)
        # x = torch.mm(mask.t(), x)
        # x = x.view(pnum, pnum, -1)
        # x = x.unsqueeze(0)
        x_input = x
        
        x_D = self.conv_D(x_input)

        x = self.conv1_forward(x_D)
        x = F.relu(x)
        x = self.conv2_forward(x)
        x = F.relu(x)
        x = self.conv3_forward(x)
        x = F.relu(x)
        x_forward = self.conv4_forward(x)

        # soft-thresholding block
        x_st = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.Sp(soft_thr)))

        x = self.conv1_backward(x_st)
        x = F.relu(x)
        x = self.conv2_backward(x)
        x = F.relu(x)
        x = self.conv3_backward(x)
        x = F.relu(x)
        x_backward = self.conv4_backward(x)

        x_G = self.conv_G(x_backward)

        # prediction output (skip connection); non-negative output
        x_pred = F.relu(x_input + x_G)

        # compute symmetry loss
        x = self.conv1_backward(x_forward)
        x = F.relu(x)
        x = self.conv2_backward(x)
        x = F.relu(x)
        x = self.conv3_backward(x)
        x = F.relu(x)
        x_D_est = self.conv4_backward(x)
        symloss = x_D_est - x_D

        return [x_pred, symloss, x_st]

# define basic block of FISTA-Net
class  UnetStyleBasicBlock(nn.Module):
    """docstring for  UnetStyleBasicBlock"""

    def __init__(self, features=16):
    # def __init__(self, features=32):
        super(UnetStyleBasicBlock, self).__init__()
        #self.lambda_step = nn.Parameter(torch.Tensor([0.2]))
        #self.soft_thr = nn.Parameter(torch.Tensor([0.05]))
        bilinear = True
        self.Sp = nn.Softplus()

        self.conv_D = nn.Conv2d(1, features, (3,3), stride=1, padding=1)

        self.conv1_forward = LightDoubleConv(features, 32)
        self.conv2_forward = Down(32, 64)
        self.conv3_forward = Up(32+64, 32, bilinear)
        self.conv4_forward = OutConv(32, features)

        factor = 2 if bilinear else 1

        self.conv1_backward = LightDoubleConv(features, 32)
        self.conv2_backward = Down(32, 64)
        self.conv3_backward = Up(32+64, 32, bilinear)
        self.conv4_backward = OutConv(32, features)

        self.conv_G = nn.Conv2d(features, 1, (3,3), stride=1, padding=1)


    def forward(self, x, fft_forback, mask, PhiTb, lambda_step, soft_thr):
        """_summary_

        Args:
            x (_type_): is rec_u_img
            PhiTPhi (_type_): _description_
            PhiTb (_type_): _description_
            mask (_type_): _description_
            lambda_step (_type_): _description_
            soft_thr (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = x - self.Sp(lambda_step) * fft_forback(x, mask) + self.Sp(lambda_step) * PhiTb
        x_input = x
        
        x_D = self.conv_D(x_input)

        x1= self.conv1_forward(x_D)
        # x = F.relu(x)
        x2 = self.conv2_forward(x1)
        # x = F.relu(x)
        x3 = self.conv3_forward(x2, x1)
        # x = F.relu(x)
        x_forward = self.conv4_forward(x3)

        # soft-thresholding block
        x_st = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.Sp(soft_thr)))

        x1 = self.conv1_backward(x_st)
        # x = F.relu(x)
        x2 = self.conv2_backward(x1)
        # x = F.relu(x)
        x3 = self.conv3_backward(x2, x1)
        # x = F.relu(x)
        x_backward = self.conv4_backward(x3)

        x_G = self.conv_G(x_backward)

        # prediction output (skip connection); non-negative output
        x_pred = F.relu(x_input + x_G)
        # x_pred = x_input + x_G

        # compute symmetry loss
        x1 = self.conv1_backward(x_forward)
        # x = F.relu(x)
        x2 = self.conv2_backward(x1)
        # x = F.relu(x)
        x3 = self.conv3_backward(x2, x1)
        # x = F.relu(x)
        x_D_est = self.conv4_backward(x3)
        symloss = x_D_est - x_D

        return [x_pred, symloss, x_st]


class SimpleCNN(nn.Module):
    ''' DnCNN with Residue Structure'''
    def __init__(self, depth=6, n_channels=8, in_chan=2, out_chan=2, add_bias=True):
        super(SimpleCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=in_chan, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.PReLU(n_channels,init=0.025))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, 
                kernel_size=kernel_size, padding=padding, bias=True))
            layers.append(nn.PReLU(n_channels,init=0.025))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=out_chan, 
            kernel_size=kernel_size, padding=padding, bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

class FISTANetPlus(nn.Module):
    # def __init__(self, LayerNo, Phi, Wt, mask):
    def __init__(self, LayerNo):
        super(FISTANetPlus, self).__init__()
        self.LayerNo = LayerNo # fista-net is 7, ista-net is 9
        # self.Phi = Phi
        # self.Wt = Wt     # learned weight from LISTA
        # self.mask =mask
        onelayer = []

        self.bb = BasicBlock()
        # self.bb = UnetStyleBasicBlock()
        for i in range(LayerNo):
            onelayer.append(self.bb)

        self.fcs = nn.ModuleList(onelayer)
        self.fcs.apply(initialize_weights)
        
        # thresholding value
        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([-2]))
        # gradient step
        self.w_mu = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu = nn.Parameter(torch.Tensor([0.1]))
        # two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([0.5]))
        self.b_rho = nn.Parameter(torch.Tensor([0]))

        self.Sp = nn.Softplus()
        self.fft_forback = FFT_Mask_ForBack()

        self.simplecnn = SimpleCNN()

    def forward(self, PhiTb, mask):
        """
        Phi   : system matrix; default dim 104 * 3228;
        mask  : mask matrix, dim 3228 * 4096
        b     : measured signal vector; u_k
        x0    : initialized x with Laplacian Reg. u_img
        """
        # convert data format from (batch_size, channel, vector_row, vector_col) to (vector_row, batch_size)
        # b = torch.squeeze(b, 1)
        # b = torch.squeeze(b, 2)
        # b = b.t()

        # LISTA
        # PhiTPhi = self.Wt.t().mm(self.Phi)
        # PhiTb = self.Wt.t().mm(b)

        # initialize the result
        x0 = PhiTb

        xold = x0
        y = xold 
        layers_sym = []     # for computing symmetric loss
        layers_st = []      # for computing sparsity constraint
        # xnews = []       # iteration result
        for i in range(self.LayerNo):
            theta_ = self.w_theta * i + self.b_theta  # soft_thr
            mu_ = self.w_mu * i + self.b_mu  # lambda_step
            y = y + self.simplecnn(y) # ! metainvnet
            [xnew, layer_sym, layer_st] = self.fcs[i](y, self.fft_forback, mask, PhiTb, mu_, theta_)
            #  (self, x, fft_forback, mask, PhiTb, lambda_step, soft_thr)
            rho_ = (self.Sp(self.w_rho * i + self.b_rho) -  self.Sp(self.b_rho)) / self.Sp(self.w_rho * i + self.b_rho)
            y = xnew + rho_ * (xnew - xold) # two-step update
            xold = xnew
            # xnews.append(xnew)   # iteration result
            layers_st.append(layer_st)
            layers_sym.append(layer_sym)

        return [xnew, layers_sym, layers_st]


def l1_loss(pred, target, l1_weight):
    """
    Compute L1 loss;
    l1_weigh default: 0.1
    """
    err = torch.mean(torch.abs(pred - target))
    err = l1_weight * err
    return err

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss. 0.01 default.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss

def train_loss(pred, y_target):
    return nn.MSELoss()(pred, y_target)
    # return nn.SmoothL1Loss()(pred, y_target)

pytorch_ssim_loss = ssim_loss.SSIM(window_size=11)

def fista_net_loss(pred, y_target, loss_layers_sym, loss_st):

    # Compute loss, data consistency and regularizer constraints
    loss_discrepancy = train_loss(pred, y_target) + l1_loss(pred, y_target, 0.1) - pytorch_ssim_loss(pred, y_target) + 1
    loss_constraint = 0
    for k, _ in enumerate(loss_layers_sym, 0):
        loss_constraint += torch.mean(torch.pow(loss_layers_sym[k], 2))
    
    sparsity_constraint = 0
    for k, _ in enumerate(loss_st, 0):
        sparsity_constraint += torch.mean(torch.abs(loss_st[k]))
    
    # loss = loss_discrepancy + gamma * loss_constraint
    loss = loss_discrepancy +  0.01 * loss_constraint + 0.001 * sparsity_constraint
    return loss

def train_l1_loss(pred, y_target):
    # return nn.MSELoss()(pred, y_target)
    return nn.SmoothL1Loss()(pred, y_target)

def fista_net_loss_l1(pred, y_target, loss_layers_sym, loss_st):

    # Compute loss, data consistency and regularizer constraints
    # loss_discrepancy = train_l1_loss(pred, y_target) + l1_loss(pred, y_target, 0.1)
    # loss_discrepancy = train_loss(pred, y_target) + l1_loss(pred, y_target, 0.1)
    loss_discrepancy = train_l1_loss(pred, y_target)
    
    loss_constraint = 0
    for k, _ in enumerate(loss_layers_sym, 0):
        loss_constraint += torch.mean(torch.pow(loss_layers_sym[k], 2))
    
    sparsity_constraint = 0
    for k, _ in enumerate(loss_st, 0):
        sparsity_constraint += torch.mean(torch.abs(loss_st[k]))
    
    # loss = loss_discrepancy + gamma * loss_constraint
    # loss = loss_discrepancy +  0.01 * loss_constraint + 0.001 * sparsity_constraint
    loss = loss_discrepancy +  0.01 * loss_constraint
    return loss
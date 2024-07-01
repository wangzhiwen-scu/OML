from torch import nn
import torch

import sys

sys.path.append('.')
from model_backbones.nafnet import NAFBlock

def Sequential(cnn, norm, ac, bn=True):
    if bn:
        return nn.Sequential(cnn, norm, ac)
    else:
        return nn.Sequential(cnn, ac)
        
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class ReconstructionForwardUnit(nn.Module):
    def __init__(self, bn):
        super(ReconstructionForwardUnit, self).__init__()
        self.conv1 = Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            # nn.BatchNorm2d(32),
            LayerNorm2d(32),
            SimpleGate())
        self.conv2 = Sequential(
            nn.Conv2d(32, 64, 3, padding=1), 
            # nn.BatchNorm2d(64),
            LayerNorm2d(64),
            SimpleGate())
        self.conv3 = Sequential(
            nn.Conv2d(64, 128, 3, padding=1), 
            # nn.BatchNorm2d(128),
            LayerNorm2d(128),
            SimpleGate())
        self.conv4 = Sequential(
            nn.Conv2d(128, 256, 3, padding=1), 
            # nn.BatchNorm2d(256),
            LayerNorm2d(256),
            SimpleGate())
        self.conv5 = Sequential(
            nn.Conv2d(256, 128, 3, padding=1), 
            # nn.BatchNorm2d(128),
            LayerNorm2d(128),
            SimpleGate())
        self.conv6 = Sequential(
            nn.Conv2d(128, 64, 3, padding=1), 
            # nn.BatchNorm2d(64),
            LayerNorm2d(64),
            SimpleGate())
        self.conv7 = Sequential(
            nn.Conv2d(64, 32, 3, padding=1), 
            # nn.BatchNorm2d(32),
            LayerNorm2d(32),
            SimpleGate())
        self.conv8 = nn.Conv2d(32, 1, 3, padding=1)
        # self.ac8 = nn.LeakyReLU(inplace=True)
        self.ac8 = SimpleGate()


    def forward(self, *input):
        x, u_x = input
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        output = self.ac8(x8 + u_x)
        return output

class FeatureExtractor(nn.Module):
    def __init__(self, bn, nafblock):
        super(FeatureExtractor, self).__init__()
        ############################################################
        # self.kspace_extractor = FeatureResidualUnit()
        # self.image_extractor = FeatureResidualUnit()

        ###########################################################
        if nafblock:
            self.kspace_extractor = FeatureForwardUnit_NAFBLOCK()
            self.image_extractor = FeatureForwardUnit_NAFBLOCK()
           
        else:
            self.kspace_extractor = FeatureForwardUnit(bn=bn, nafblock=nafblock)
            self.image_extractor = FeatureForwardUnit(bn=bn, nafblock=nafblock)

        ############################################################

        initialize_weights(self)

    def forward(self, *input):
        k, img = input
        k_feature = self.kspace_extractor(k)
        img_feature = self.image_extractor(img)

        return k_feature, img_feature

class FeatureForwardUnit(nn.Module):
    def __init__(self, negative_slope=0.01, bn=True, nafblock=False):
        super(FeatureForwardUnit, self).__init__()
        self.nafblock = nafblock
        if nafblock:
            self.conv1 = Sequential(
                nn.Conv2d(2, 32, 3, padding=1),
                # nn.BatchNorm2d(32),
                # nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
                LayerNorm2d(32),
                # SimpleGate()
                nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
                
            self.conv2 = Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                LayerNorm2d(32),
                # SimpleGate())
                nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
            self.conv3 = Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                LayerNorm2d(32),
                # SimpleGate())
                nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
            self.conv4 = Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                LayerNorm2d(32),
                # SimpleGate())
                nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
            # self.conv5 = Sequential(
            #     nn.Conv2d(32, 32, 3, padding=1),
            #     nn.BatchNorm2d(32),
            #     nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
            self.conv6 = nn.Conv2d(32, 2, 3, padding=1)
            # self.ac6 = SimpleGate()
            self.ac6 = nn.LeakyReLU(negative_slope=negative_slope)

        else:
            self.conv1 = Sequential(
                nn.Conv2d(2, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
            self.conv2 = Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
            self.conv3 = Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
            self.conv4 = Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
            # self.conv5 = Sequential(
            #     nn.Conv2d(32, 32, 3, padding=1),
            #     nn.BatchNorm2d(32),
            #     nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
            self.conv6 = nn.Conv2d(32, 2, 3, padding=1)
            self.ac6 = nn.LeakyReLU(negative_slope=negative_slope)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        # out5 = self.conv5(out4)
        out6 = self.conv6(out4)

        
        # if self.nafblock:
        output = self.ac6(out6) + x
        # else:
            # output = self.ac6(out6 + x)

        return output
    

class FeatureForwardUnit_NAFBLOCK(nn.Module):
    def __init__(self, negative_slope=0.01, bn=True, nafblock=False, img_channel=2, width=16):
        super(FeatureForwardUnit_NAFBLOCK, self).__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        chan = width
        self.nafbolck = NAFBlock(chan)

    def forward(self, x_o):
        x = self.intro(x_o)
        x = self.nafbolck(x)
        output = self.ending(x)
        # output = output + x_o
        return output
''' Genereal network structure with Dn-CNN, Db-INV and P-FCN '''
import sys

sys.path.append('.')
from ct_dependencies.EI_dependencies.ct import CT

import torch
from torch import nn
# from utils import aver_psnr,aver_ssim_tr
from metainvnet.utils import torch_W,torch_Wt
import numpy as np

import torch.nn.functional as F

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetaInvH(nn.Module):
    '''MetaInvNet with heavy weight CG-Init'''
    def __init__(self):
        super(MetaInvH, self).__init__()
        
    def forward(self, CgModule, sino, x, laam, miu, CGInitCNN):
        Wu=CgModule.W(x)
        dnz=F.relu(Wu-laam)-F.relu(-Wu-laam)
        PtY=CgModule.BwAt(sino)
        muWtV=CgModule.Wt(dnz)
        rhs=PtY+muWtV*miu

        uk0=CGInitCNN(x)
        Ax0=CgModule.AWx(uk0,miu)
        res=Ax0-rhs
        img=CG_alg(CgModule, uk0, miu, res, CGiter=5)
        return img


def CG_alg(myAtA,x,mu,res,CGiter=20):
    r=res
    p=-res
    for k in range(CGiter):
        pATApNorm = myAtA.pATAp(p)
        mu_pWtWpNorm=myAtA.pWTWp(p,mu)
        rTr=torch.sum(r**2,dim=(1,2,3))
        alphak = rTr/(mu_pWtWpNorm+pATApNorm)
        alphak = alphak.view(-1,1,1,1)
        x = x+alphak*p
        r = r+alphak*myAtA.AWx(p,mu)
        betak = torch.sum(r**2,dim=(1,2,3))/ rTr
        betak = betak.view(-1,1,1,1)
        p=-r+betak*p

    pATApNorm = myAtA.pATAp(p)
    mu_pWtWpNorm=myAtA.pWTWp(p,mu)
    rTr=torch.sum(r**2,dim=(1,2,3))
    alphak = rTr/(mu_pWtWpNorm+pATApNorm)
    alphak = alphak.view(-1,1,1,1)
    x = x+alphak*p
    return x

# ! main1
class MetaInvNet():
    def __init__(self, args, options=None):

        optionss = options.detach().cpu().numpy()
        img_hw = int(optionss[2])
        n_views = int(optionss[0])
        self.physics = CT(img_hw, n_views, circle=False, device=device)
        self.projector = self.physics.A
        self.projector_t = self.physics.A_dagger
        self.radon = self.projector
        self.iradon = self.projector_t
        self.op_norm = None

        self.layers = args.layers
        self.net = MetaInvNet_H(args, InitNet=MetaInvH)
        # self.net = MetaInvL(args, InitNet=MetaInvH)
        self.args = args
        self.net = self.net.cuda()

    def tr_model(self, sino, u0):
        self.net.train()
        sino.requires_grad = True
        u0.requires_grad = True
        recU= self.net(sino, u0, self.radon, self.iradon, self.op_norm)
        # recU is a list of length 4. we take last one.
        return recU

class CGClass():
    def __init__(self,Radon,iRadon,op_norm, WDec,WRec):
        self.FwA=Radon
        self.BwAt=iRadon
        self.WDec=WDec
        self.WRec=WRec
        self.op_norm=op_norm
    def AWx(self,img,mu):
        Ax0=self.BwAt(self.FwA(img))+self.Wt(self.W(img))*mu
        return Ax0
    def W(self,img):
        return self.WDec(img)
    def Wt(self,Wu):
        return self.WRec(Wu)
    def pATAp(self,img):
        Ap=self.FwA(img)
        AtAp=self.BwAt(Ap)
        pATApNorm=torch.sum(img*AtAp,dim=(1,2,3))
        return pATApNorm
    def pWTWp(self,img,mu):
        Wp=self.W(img)
        mu_Wp=mu*Wp*Wp
        pWTWpNorm=torch.sum(mu_Wp,dim=(1,2,3))
        return pWTWpNorm


class InitCG(nn.Module):
    ''' DnCNN with Residue Structure'''
    def __init__(self, depth=17, n_channels=64, in_chan=1, out_chan=1,add_bias=True):
        super(InitCG, self).__init__()
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


class MetaInvL(nn.Module):
    '''MetaInvNet with light weight CG-Init'''
    def __init__(self):
        super(MetaInvL, self).__init__()
        self.CGInitCNN=InitCG(depth=6, n_channels=8, in_chan=1, out_chan=1, add_bias=True)
        
    def forward(self, CgModule, sino, x,laam,miu):
        Wu=CgModule.W(x)
        dnz=F.relu(Wu-laam)-F.relu(-Wu-laam)
        PtY=CgModule.BwAt(sino)
        muWtV=CgModule.Wt(dnz)*miu
        rhs=PtY+muWtV

        uk0=x+self.CGInitCNN(x)
        Ax0=CgModule.AWx(uk0,miu)
        res=Ax0-rhs
        db=CG_alg(CgModule, uk0, miu, res, CGiter=5)
        return db,uk0

# ! main2 
from .unet_model import UNet
class MetaInvNet_H(nn.Module):
    def __init__(self, args, InitNet = MetaInvH):
        super(MetaInvNet_H,self).__init__()
        self.args = args
        self.net = nn.ModuleList()
        self.net = self.net.append(InitNet())
        for i in range(args.layers):
            self.net = self.net.append(InitNet())
        self.CGInitCNN=UNet(n_channels=1, n_classes=1)

    def forward(self, sino, fbpu, radon, iradon, op_norm):
        img_list = [None] * (self.args.layers + 1)
        CgModule = CGClass(radon, iradon, op_norm, torch_W, torch_Wt)
        
        laam=torch.tensor(np.array(0.005)).float()
        miu=torch.tensor(np.array(0.01)).float()
        img_list[0] = self.net[0](CgModule, sino.detach(), fbpu.detach(),laam, miu, self.CGInitCNN)
        inc_lam, inc_miu=0.0008, 0.02 
        for i in range(self.args.layers-1):
            laam=laam-inc_lam
            miu=miu+inc_miu
            img_list[i+1] = self.net[i+1](CgModule, sino.detach(), img_list[i],laam, miu, self.CGInitCNN)

        i = self.args.layers-1
        laam=laam-inc_lam 
        miu=miu+inc_miu 
        img_list[i+1] = self.net[i+1](CgModule, sino.detach(), img_list[i],laam,miu, self.CGInitCNN)
        return img_list


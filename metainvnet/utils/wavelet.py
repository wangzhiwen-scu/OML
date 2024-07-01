''' wavelet toolbox for multilevel wavelet decomposition and reconstruction '''
'''
PyTorch Tight wavelet frame transform
Author: Haimiao Zhang
Email: hmzhang@163.com
'''
import numpy as np
import torch 

def GenerateFrameletFilter(frame):
    # Haar Wavelet
    if frame==0:
        D1=np.array([0.0, 1.0, 1.0] )/2
        D2=np.array([0.0, 1, -1])/2
        D3=('cc')
        R1=np.array([1 , 1 ,0])/2
        R2=np.array([-1, 1, 0])/2
        R3=('cc')
        D=[D1,D2,D3]
        R=[R1,R2,R3]
    # Piecewise Linear Framelet
    elif frame==1:
        D1=np.array([1.0, 2, 1])/4
        D2=np.array([1, 0, -1])/4*np.sqrt(2)
        D3=np.array([-1 ,2 ,-1])/4
        D4='ccc';
        R1=np.array([1, 2, 1])/4
        R2=np.array([-1, 0, 1])/4*np.sqrt(2)
        R3=np.array([-1, 2 ,-1])/4
        R4='ccc'
        D=[D1,D2,D3,D4]
        R=[R1,R2,R3,R4]
    # Piecewise Cubic Framelet
    elif frame==3:
        D1=np.array([1, 4 ,6, 4, 1])/16
        D2=np.array([1 ,2 ,0 ,-2, -1])/8
        D3=np.array([-1, 0 ,2 ,0, -1])/16*np.sqrt(6)
        D4=np.array([-1 ,2 ,0, -2, 1])/8
        D5=np.array([1, -4 ,6, -4, 1])/16
        D6='ccccc'
        R1=np.array([1 ,4, 6, 4 ,1])/16
        R2=np.array([-1, -2, 0, 2, 1])/8
        R3=np.array([-1, 0 ,2, 0, -1])/16*np.sqrt(6)
        R4=np.array([1 ,-2, 0, 2, -1])/8
        R5=np.array([1, -4, 6, -4 ,1])/16
        R6='ccccc'
        D=[D1,D2,D3,D4,D5,D6]
        R=[R1,R2,R3,R4,R5,R6]
    return D,R


D,R=GenerateFrameletFilter(frame=1)
D_tmp=torch.zeros(3,1,3,1)
for ll in range(3):
    D_tmp[ll,]=torch.from_numpy(np.reshape(D[ll],(-1,1)))

W=D_tmp
W2=W.permute(0,1,3,2)
kernel_dec=np.kron(W.numpy(),W2.numpy())
kernel_dec=torch.tensor(kernel_dec,requires_grad=False,dtype=torch.float32)

R_tmp=torch.zeros(3,1,1,3)
for ll in range(3):
    R_tmp[ll,]=torch.from_numpy(np.reshape(R[ll],(1,-1)))

R=R_tmp
R2=R_tmp.permute(0,1,3,2)
kernel_rec=np.kron(R2.numpy(),R.numpy())
kernel_rec=torch.tensor(kernel_rec,requires_grad=False,dtype=torch.float32).view(1,9,3,3)

import torch.nn.functional as F

def torch_W(img, kernel_dec=kernel_dec.cuda()):
    Dec_coeff=F.conv2d(F.pad(img, (1,1,1,1), mode='circular'), kernel_dec[1:,...])
    return Dec_coeff

def torch_Wt(Dec_coeff,kernel_rec=kernel_rec.cuda()):
    kernel_rec=kernel_rec.view(9,1,3,3)
    tem_coeff=F.conv2d(F.pad(Dec_coeff, (1,1,1,1), mode='circular'), kernel_rec[1:,:,...],groups=8)
    rec_img=torch.sum(tem_coeff,dim=1,keepdim=True)
    return rec_img

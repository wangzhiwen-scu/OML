import os
from glob import glob
import numpy as np
import torch

import odl
from skimage import transform
from odl.contrib import torch as odl_torch
import pydicom
import scipy.io as sio 


class Phantom2dDataset():
    def __init__(self, args, phase, datadir):
        self.phase  = phase
        self.img_size  = args.img_size
        self.sino_size  = args.sino_size
        self.args   = args

        if self.phase == 'tr':
            self.sp_file = glob(datadir)

    def  __len__(self):
        if self.phase == 'tr':
            img_num = len(self.sp_file)

        return img_num

    def radon_op(self):
        xx=200
        space = odl.uniform_discr([-xx, -xx], [xx, xx], [self.args.img_size[0], self.args.img_size[0]],dtype='float32')
        angles=np.array(self.args.sino_size[0]).astype(int)
        angle_partition = odl.uniform_partition(0, 2 * np.pi, angles)
        detectors=np.array(self.args.sino_size[1]).astype(int)
        detector_partition = odl.uniform_partition(-480,480, detectors)
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,src_radius=600, det_radius=290)
        operator = odl.tomo.RayTransform(space, geometry,impl='astra_cuda')
        
        op_norm=odl.operator.power_method_opnorm(operator)
        op_norm=torch.from_numpy(np.array(op_norm*2*np.pi)).double().cuda()

        op_layer = odl_torch.operator.OperatorModule(operator)
        op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
        fbp = odl.tomo.fbp_op(operator,filter_type='Ram-Lak',frequency_scaling=0.9)*np.sqrt(2)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

    def get_items(self, ii,radon,fbp):
        '''
        load training item one by one
        '''
        if 'dcm' in self.sp_file[ii]:
            decide_data_type='dcm'

            dcm=pydicom.read_file(self.sp_file[ii])
            dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            data=dcm.image
            data=np.array(data).astype(float)
            data=transform.resize(data, (self.img_size))
            phantom = np.rot90(data, 0)
            phantom=(phantom-np.min(phantom))/(np.max(phantom)-np.min(phantom))

        phantom=torch.from_numpy(phantom)
        phantom = phantom.unsqueeze(0)
        sino=radon(phantom)

        # add Poisson noise
        intensityI0=self.args.poiss_level
        scale_value=torch.from_numpy(np.array(intensityI0).astype(np.float))
        normalized_sino=torch.exp(-sino/sino.max())
        th_data=np.random.poisson(scale_value*normalized_sino)
        sino_noisy=-torch.log(torch.from_numpy(th_data)/scale_value)
        sino_noisy = sino_noisy*sino.max()

        # add Gaussian noise
        noise_std=self.args.gauss_level
        noise_std=np.array(noise_std).astype(np.float)
        nx,ny=np.array(self.args.sino_size[0]).astype(np.int),np.array(self.args.sino_size[1]).astype(np.int)
        noise = noise_std*np.random.randn(nx,ny)
        noise = torch.from_numpy(noise)
        sino_noisy = sino_noisy + noise

        fbpu=fbp(sino_noisy)
        phantom=phantom.type(torch.DoubleTensor)
        fbpu=fbpu.type(torch.DoubleTensor)
        sino_noisy=sino_noisy.type(torch.DoubleTensor)

        return phantom, fbpu, sino_noisy

    def get_test_item(self, ik,radon,fbp,iradon):
        '''
        load test item one by one
        '''
        if 'dcm' in self.sp_file[ik]:
            decide_data_type='dcm'   
            dcm=pydicom.read_file(self.sp_file[ik])
            dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            data=dcm.image
            data=np.array(data).astype(float)
            phantom=transform.resize(data, (self.img_size))
            phantom=(phantom-np.min(phantom))/(np.max(phantom)-np.min(phantom))

        phantom=torch.from_numpy(phantom)
        phantom = phantom.unsqueeze(0)
        sino=radon(phantom)

        # add Poisson noise
        intensityI0=self.args.test_poiss_level
        scale_value=torch.from_numpy(np.array(intensityI0).astype(np.float))
        normalized_sino=torch.exp(-sino/sino.max())
        th_data=np.random.poisson(scale_value*normalized_sino)
        sino_noisy=-torch.log(torch.from_numpy(th_data)/scale_value)
        sino_noisy = sino_noisy*sino.max()

        # add Gaussian noise
        noise_std=self.args.test_gauss_level
        noise_std=np.array(noise_std).astype(np.float)
        nx,ny=np.array(self.args.sino_size[0]).astype(np.int),np.array(self.args.sino_size[1]).astype(np.int)
        noise = noise_std*np.random.randn(nx,ny)
        noise = torch.from_numpy(noise)
        sino_noisy = sino_noisy + noise

        fbpu=fbp(sino_noisy)
        phantom=phantom.type(torch.DoubleTensor)
        fbpu=fbpu.type(torch.DoubleTensor) 
        sino_noisy=sino_noisy.type(torch.DoubleTensor)

        return phantom, fbpu, sino_noisy

    def get_imgs(self):
        img_num = self.__len__()
        radon, iradon, fbp, op_norm = self.radon_op()

        img = np.zeros((img_num, 1, *self.img_size), dtype=np.float64)
        u0 = np.zeros((img_num, 1, *self.img_size), dtype=np.float64)
        nx,ny=np.array(self.sino_size[0]).astype(np.int),np.array(self.sino_size[1]).astype(np.int)
        sino = np.zeros((img_num, 1, nx,ny), dtype=np.float64)
        kk = 0
        for ii in range(img_num):
            img[kk,0,], u0[kk,0,], sino[kk,0,] = self.get_items(ii,radon,fbp)
            kk += 1

        img = torch.from_numpy(img).to(torch.float32)
        u0 = torch.from_numpy(u0).to(torch.float32)
        sino = torch.from_numpy(sino).to(torch.float32)
        return img, u0, sino

import sys
import glob
# import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data import Dataset
# import ctlib
from torch.autograd import Function
import math
import ctlib as ctlib_v2
import numpy
import random

sys.path.append('.') # 自己的py文件所在的文件夹路径，该放__init__.py的就放
from ct_dependencies.learn import LEARN
from utils.visdom_visualizer import VisdomLinePlotter


systemMat = torch.sparse.FloatTensor()
cuda = True if torch.cuda.is_available() else False

# class fan_ed(Function):
#     def __init__(self, views, dets, width, height, dImg, dDet, dAng, s2r, d2r, binshift):
#         self.options = torch.Tensor([views, dets, width, height, dImg, dDet, dAng, s2r, d2r, binshift])
#         self.options = self.options.double().cuda()
#
#     def forward(self, image):
#         return ctlib_v2.projection(image, self.options, 0)
#
# class trainset_loader(Dataset):
#     def __init__(self):
#         self.files_A = sorted(glob.glob('train/label/data' + '*.mat'))
#
#     def __getitem__(self, index):
#         file_A = self.files_A[index]
#         # file_B = 'geometry_5/' + file_A.replace('label','projection')
#         file_B = file_A.replace('label','projection')
#         input_data = sio.loadmat(file_A)['data']
#         input_data = torch.FloatTensor(input_data.reshape(1,256,256))
#         return input_data, file_B
#
#     def __len__(self):
#         return len(self.files_A)
#
# class testset_loader(Dataset):
#     def __init__(self):
#         self.files_A = sorted(glob.glob('test/label/data' + '*.mat'))
#
#     def __getitem__(self, index):
#         file_A = self.files_A[index]
#         file_B = 'geometry_9/' + file_A.replace('label','projection')
#         input_data = sio.loadmat(file_A)['data']
#         input_data = torch.FloatTensor(input_data.reshape(1,256,256))
#         return input_data, file_B
#
#     def __len__(self):
#         return len(self.files_A)

# self.gemoetry = torch.FloatTensor([
#     [512, 368, 256, 256, 0.0133, 0.025716, 0.012268, 5.95, 4.906, 0, 0.5e5],
#     [512, 315, 256, 256, 0.014, 0.03, 0.012268, 4.5, 3.5, 0, 5e5 * 0.1375],
#     [384, 330, 256, 256, 0.0139, 0.026, 0.0164, 4, 3, 0, 5e5 * 0.175],
#     [400, 350, 256, 256, 0.012, 0.022, 0.0157, 4, 3.5, 0, 5e5 * 0.2125],
#     [384, 350, 256, 256, 0.014, 0.025, 0.0164, 5, 3, 0, 5e5 * 0.25]
# ])

# https://github.com/xiawj-hub/CTLIB 

class validate_loader(Dataset):
    def __init__(self):
        # self.files_A = sorted(glob.glob('validate/label/data' + '*.mat'))   ##src_path
        # self.files_A = sorted(glob.glob('D:/dataset/meta_learning/large-train/geometry_5/train/label/data' + '*.mat')) ## Test dataset
        # self.files_A = sorted(glob.glob('./genar_data/geometry_5/test/label/' + '*.mat'))
        self.files_A = sorted(glob.glob('./ct_dependencies/' + '*.mat'))

    def __getitem__(self, index):
        ## bone is 17
        ## fuqiang is 8
        file_A = self.files_A[index]
        file_B = file_A.replace('label','projection')
        ## 1 View 2 Detector 3 image_height 4 image_weight 5 physical_pixel 6 distance_det
        ### 7 distance_view 8 s2r 9 d2r 10 binshift
        gen_bias_1 = random.randint(-100,100) * 0.01
        gen_bias_2 = random.randint(-100,100) * 0.01

        # options =  [384, 350, 256, 256, 0.014, 0.025, 0.0164, 5 + gen_bias_1, 3 + gen_bias_2, 0, 5e5 * 0.25, 0]
        options =  [384, 350, 256, 256, 0.014, 0.025, 0.0164, 5 + gen_bias_1, 3 + gen_bias_2, 0, 5e5 * 0.25]

        # options =  [512, 736, 256, 256, 0.006641, 0.012858, 0, 0.012268, 5.95, 4.906, 0, 0] # 💎 512
        options =  [60, 736, 256, 256, 0.006641, 0.012858, 0, 0.104719, 5.95, 4.906, 0, 0] # 💎 512

        # 2*pi / views =dAng
        # 2*pi / views = dAng
    
        # options = [400, 350, 256, 256, 0.012, 0.022, 0.0157, 4 + gen_bias_1, 3.5 + gen_bias_2, 0, 5e5 * 0.2125];
        # options = [384, 330, 256, 256, 0.0139, 0.026, 0.0164, 4 + gen_bias_1, 3 + gen_bias_2, 0, 5e5 * 0.175];
        # options = [512, 315, 256, 256, 0.014, 0.03, 0.012268, 4.5 + gen_bias_1, 3.5 + gen_bias_2, 0, 5e5 * 0.1375]
        # options = [512, 368, 256, 256, 0.0133, 0.025716, 0.012268, 5.95 + gen_bias_1, 4.906 + gen_bias_2, 0, 0.5e5]

        options = numpy.array(options)
        options = torch.from_numpy(options)
        input_data = sio.loadmat(file_A)['data']

        input_data = torch.FloatTensor(input_data.reshape(1,256,256))

        return input_data, file_B, options

    def __len__(self):
        return len(self.files_A)

def addPoissNoise(Pro, dose, var):
    temp = torch.poisson(dose * torch.exp(-Pro))
    # elec_noise = math.sqrt(var)*torch.randn(temp.size())
    elec_noise = torch.normal(0, math.sqrt(var), temp.size())
    elec_noise = elec_noise.cuda()
    elec_noise = torch.round(elec_noise)
    temp = temp.cuda()
    temp = temp+elec_noise
    ##
    temp= torch.clamp(temp,min=0.)

    p_noisy = -torch.log((temp/dose)+0.000001)
    ##
    # p_noisy = torch.clamp(p_noisy,min=0.)
    return p_noisy

def filtered(prj,options):
    print(options)
    dets = int(options[1])
    dDet = options[5]
    s2r = options[7].cuda()
    d2r = options[8]
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
    w = torch.arange((-dets / 2 + 0.5) * virdet, dets / 2 * virdet, virdet).cuda()
    w = s2r / torch.sqrt(s2r ** 2 + w ** 2)
    w = w.view(1, 1, 1, -1).cuda()
    filter = filter.view(1, 1, 1, -1).cuda().double()
    # self.options = nn.Parameter(options, requires_grad=False)
    coef = pi / options[0]
    p = prj * virdet * w * coef
    p = torch.nn.functional.conv2d(p, filter, padding=(0, dets - 1))
    p = p.squeeze()
    return p

if __name__ == "__main__":
    dose = 0.5e5
    var = 10

    validate_data = DataLoader(validate_loader(), batch_size=1, shuffle=False, num_workers=2)

    # options = [128, 512, 256, 256, 0.007, 0.0075, 0.0491, 4.0, 3.0, 0, 5e5]
    # projector = fan_ed(options)
    # # a = numpy.array(options)
    # options = [512, 368, 256, 256, 0.0133, 0.025716, 0.012268, 5.95, 4.906, 0, 0.5e5]
    #
    # options = numpy.array(options)
    # options = torch.from_numpy(options)

    path = './test/'
    plotter = VisdomLinePlotter(env_name='test')

    for batch_index, data in enumerate(validate_data):
        input_data, file_name, options = data
        #####  TEST SET
        options=options.squeeze()
        learn = LEARN(options).cuda().double()

        input_data = (input_data - torch.min(input_data) )/ (torch.max(input_data)-torch.min(input_data))

        # input_data = input_data/5.8522
        # input_data = input_data / 3.84
    #####  TEST SET
    
        input_data = input_data.double()
        if cuda:
            input_data = input_data.cuda()
        temp = []

        options = options.cuda()

        # print(input_data.shape)

        plotter.image('input_data', 1, input_data)

        proj = ctlib_v2.projection(input_data, options)

        plotter.image('proj', 1, proj)

        # proj_possnoise=filtered(addPoissNoise(proj, dose,var),options)
        # plotter.image('proj_possnoise', 1, proj_possnoise)
        # print(proj.shape)

        fbpdata = ctlib_v2.fbp(proj, options)
        plotter.image('fbpdata', 1, fbpdata)


        out = learn(fbpdata, proj)
        # https://github.com/xiawj-hub/Physics-Model-Data-Driven-Review/blob/main/recon/models/LEARN.py
        # change projection_t into fbp
        plotter.image('out', 1, out)

        

        # proj_cur=proj.detach().squeeze().cpu().numpy()
        # input_data_cur=input_data.detach().squeeze().cpu().numpy()
        # fbpdata_cur = fbpdata.detach().squeeze().cpu().numpy()
        # out=out.detach().squeeze().cpu().numpy()
        # sio.savemat('./ct_dependencies/fbpdata.mat', {'input_data':input_data_cur, 'fbp':fbpdata_cur, 'proj':proj_cur, 'learn':out})

        # print(fbpdata.shape)

        # noised_pro= []
        # filed_pro = []
        # for i in range(len(proj)):

        #    # noised_pro.append(filtered(addPoissNoise(proj[i]*5.8522,dose,var)/5.8522,options[i]))
        #    noised_pro.append(filtered(addPoissNoise(proj[i], dose, var), options[i]))
        #    # noised_pro.append(filtered(addPoissNoise(proj[i],dose,var),options[i]))
        #    # noised_pro.

        #    #  noised_pro.append(filtered(proj[i],options[i]))

        # backproj = ctlib_v2.fbpbackprojection(noised_pro, options)
        # for i in range(len(backproj)):
        #     # temp = proj[i].cpu().numpy().reshape(128,512)

        #     # name = file_name[i][-13:]
        #     # name = file_name[i][:40]
        #     name = file_name[i].replace(file_name[i][:40],'/')
        #     # print(file_name)
        #     print(name)
        #     # break
        #     # name2=file_name[i].replace(name,'./')
        #     # print(name2)
        #     geo_path = path + './test/geometry_' + name
        #     input_path = path + './test/input_' + name
        #     pro_path = path + './test/projection_' + name
        #     # print(geo_path)
        #     # break
        #     ## Save input
        #     temp = backproj[i].cpu().numpy()
        #     temp = temp.squeeze()
        #     sio.savemat(input_path, {'data':temp})

        #     ## Save Geo
        #     opt = options[i].cpu().numpy()
        #     sio.savemat(geo_path,{'data':opt})

        #     ## Save proj
        #     proj_ = proj[i].cpu().numpy()
        #     temm = proj_.squeeze()
        #     sio.savemat(pro_path, {'data':temm})

        # # break
        # print(batch_index)




        # proj = projector.forward(input_data)
        # for i in range(proj.size(0)):
        #     temp = proj[i].cpu().numpy().reshape(128,512)
        #     sio.savemat(file_name[i], {'data':temp})
        #     break


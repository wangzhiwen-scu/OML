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
import SimpleITK as sitk

from torchvision.transforms import transforms
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import imageio.v3 as iio

sys.path.append('.') # 自己的py文件所在的文件夹路径，该放__init__.py的就放
from ct_dependencies.learn import LEARN
from utils.visdom_visualizer import VisdomLinePlotter

from data.ct_toolbox import CT_path
# https://github.com/xiawj-hub/Physics-Model-Data-Driven-Review/blob/main/recon/models/LEARN.py
# change projection_t into fbp

systemMat = torch.sparse.FloatTensor()
cuda = True if torch.cuda.is_available() else False

class yzy_validate_loader(Dataset):
    def __init__(self):

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

class CT_dataset(Dataset): 
    def __init__(self, dataset_name, change_isotopic=True, root=None, validation=False, test=False, slight_motion=False, one_shot=False, seed=42):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        self.dataset_name = dataset_name

        if dataset_name == 'mayo':
            train_data, test_data  = CT_path.get_mayo_path_lists()
            self.getslicefromNpy = CT_path.get_mayo_dicom

        elif dataset_name == "deeplesion":
            train_data, test_data  = CT_path.get_deeplesion_path_lists()
            self.getslicefromNpy = CT_path.get_deeplesion_dicom
        elif dataset_name == "spine":
            train_data, test_data  = CT_path.get_spine_path_lists()
            self.getslicefromNpy = CT_path.get_spine_dicom
        elif dataset_name == "cta":
            train_data, test_data  = CT_path.get_cta_path_lists()
            self.getslicefromNpy = CT_path.get_cta_dicom

        # if validation:
            # test_data = test_data_orig[0:1]
            
        if root == None:
            paired_files = train_data

        if validation:
            paired_files = test_data

        if test:
            # paired_files = test_data_orig
            pass

        self.examples = []

        for fname in paired_files:


            start_slice = 0
            step = 1
            if dataset_name == 'mayo':
                item_img = sitk.GetArrayFromImage(sitk.ReadImage(fname)) # (slice, h,w)
                num_slices = item_img.shape[0]

                start_slice = 0
                end_slice = num_slices
            
            elif dataset_name == "deeplesion":
                # item_img = iio.imread(fname)
                num_slices = 1

                start_slice = 0
                end_slice = num_slices

            elif dataset_name == 'spine':
                item_img = sitk.GetArrayFromImage(sitk.ReadImage(fname)) # (~ 103, h,w)
                num_slices = item_img.shape[0]
                start_slice = 30
                end_slice = 80
                step = 5

            elif dataset_name == 'cta':
                # item_img = sitk.GetArrayFromImage(sitk.ReadImage(fname)) # (h,w) 160~190 slices
                # num_slices = item_img.shape[0]
                start_slice = 50
                end_slice = 100
                step = 5

            # shotname = get_h5py_shotname(fname)


            shotname = '1'
            
            self.examples += [(fname, shotname, slice) for slice in range(start_slice, end_slice, step)] 

        #! writ it in one examples.... https://blog.csdn.net/wunianwn/article/details/126965641
        if test or not change_isotopic:
            self.transform = transforms.Compose([
                    transforms.ToTensor()

                ]
            ) 
            self.target_transform = transforms.Compose([
                    transforms.ToTensor()

                ]
            )           
        else:
            self.transform = transforms.Compose([
                    
                    transforms.ToTensor(),
                    # transforms.RandomRotation(5),
                    # transforms.RandomHorizontalFlip(p=0.1)
                    
                ]
            ) 
            self.target_transform = transforms.Compose([
                    
                    transforms.ToTensor(),
                    transforms.RandomRotation(5),  # csmri 15
                    transforms.RandomHorizontalFlip(p=0.1) # csmri 0.5
                    
                ]
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, shotname, slice = self.examples[i]
        img = self.getslicefromNpy(fname, slice)

        shotname = shotname+'_slice'+ str(slice)


        img = img.squeeze()
        img = img*1.0 # unit16 to float32
        if self.dataset_name == "mayo":
            img=(img) / 1000*0.192 +0.192
            # img = cv2.resize(img, [256, 256])
        elif self.dataset_name == "deeplesion":
            img = img - 32768 
            img[img < -1000] = -1000
            img=(img) / 1000*0.192 +0.192

        elif self.dataset_name == "spine":
            # spine (1680,-3024)
            img[img < -1000] = -1000
            img=(img) / 1000*0.192 +0.192
        elif self.dataset_name == "cta":
            # spine (3122,0)
            img=(img) / 1000*0.192 +0.192
        IMG_SIZE = 256
        img = cv2.resize(img, [IMG_SIZE, IMG_SIZE])

        # img = (img -img.min()) / (img.max() - img.min())

        # show clip(0.12, 0.3) 往中间调。
        # show clip(0, 0.3) 肺。

        # img = cv2.resize(img, [256, 256])

        # if self.dataset_name == "cc_data2_brain_t1" or self.dataset_name == "fastmri_brain_t2":
        #     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # elif self.dataset_name == "ixi_t1_periodic_slight_sagittal":
        #     img = cv2.rotate(img, cv2.ROTATE_180)            
        #     img = cv2.flip(img, 1) # 1 is vertically flip

        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed) 
        img = self.transform(img)

        # img_max = torch.max(torch.abs(img))
        # img = img / img_max

        # if self.dataset_name == "cc_data2_brain_t1":
        #     img = torch.abs(img)

        return img, 1, shotname


if __name__ == "__main__":
    import time

    dose = 0.5e5
    var = 10


    # todo deeplesion, 3w range
    dataset_name_lists = ['mayo', 'deeplesion', 'spine', 'cta']
    batch = 1

    # dataset_sup = CT_dataset(dataset_name=dataset_name_lists[-1])
    dataset_sup = CT_dataset(dataset_name='spine', test=False)
    
    dataloader_sup = DataLoader(dataset_sup, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)  # 增加GPU利用率稳定性。

    step = 0
    plotter = VisdomLinePlotter(env_name='test')

    options =  [5, 736, 256, 256, 0.006641*10, 0.012858*10, 0, 0.104719*10, 5.95, 4.906, 0, 0] # 💎 512
    # options =  [60, 736, 256, 256, 0.006641, 0.012858, 0, 0.104719, 5.95, 4.906, 0, 0] # 💎 512
    # options =  [60, 736, 512, 512, 0.006641, 0.012858, 0, 0.104719, 5.95, 4.906, 0, 0] # 💎 512

    options = numpy.array(options)
    options = torch.from_numpy(options).cuda().double()

    loop = tqdm(enumerate(dataloader_sup), total=len(dataloader_sup))
    for index, (img, _temp, shotname) in loop:
    # for index, x in loop:
    # for x in dataloader_sup:400
        step += 1

        # img, _temp, shotname = x

        # show clip(0.12, 0.3) 往中间调。
        # show clip(0, 0.3) 肺。

        img = img.cuda().double()
        proj = ctlib_v2.projection(img, options)
        fbpdata = ctlib_v2.fbp(proj, options)

        plotter.image('clip', step, torch.clip(img, 0.3, 0.32))

        plotter.image(shotname[0], step, img)
        plotter.image('proj', 1, proj)
        plotter.image('fbpdata', 1, fbpdata)

        time.sleep(0.1)
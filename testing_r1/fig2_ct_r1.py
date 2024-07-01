"""
    Mask Learning Module.
    Pytorch version.
    By WZW.
"""
import os
import sys
import time
import torch
import warnings

# from models.helper import ssim, psnr2
import numpy as np
from matplotlib.image import imsave
from mpl_toolkits.axes_grid1 import AxesGrid

import argparse
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.append('.') # 自己的py文件所在的文件夹路径，该放__init__.py的就放
from data.dataset import get_tta_ct_dataset

from utils.train_utils import mkdir
from utils.visdom_visualizer import VisdomLinePlotter
from testing.toolbox.test_toolbox import get_metrics, newdf2excel, dataframe_template, get_ct_map

# load models
from ct_dependencies.EI_dependencies.ct import CT

from training_ct.ct_py.EI import ModelSet as EI
from training_ct.ct_py.learn import ModelSet as LEARN
from training_ct.ct_py.ttt_tta import ModelSet as TTT
from training_ct.ct_py.metainvnet_train import ModelSet as Metainvnet
from training_ct.ct_py.our_s2_tta import ModelSet as Our

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

#######################################-TEST-#########################################################

class NameSpace(object):
    def __init__(self, args):
        #    x1,y1 -----(x2, y1)
        #    |          |
        #    |          |
        #    |          |
        # (x1, y2)-------x2,y2

        # x1, y1, x2, y2
        self.zoom_factor=1.5
        if args.shift == "anatomy":
            self.bbox = (100, 100+40, 190, 190+40)
            # cta
            

            self.hu = (0.38, 0.45)
        elif args.shift == "dataset":
            
            # self.bbox = (100, 10+40, 190, 100+40) 
            # self.bbox = (40, 40+40, 130, 120+40) 
            self.bbox = (40, 110+40, 130, 200+40) 

            self.hu = (0.12, 0.35)
            # self.hu = (-0.03, 0.04)
        # elif args.shift == "modality":
            # self.bbox = (80, 30+40, 150, 100+40) 
            
        elif args.shift == "n_views":
            self.bbox = (100, 100+40, 190, 190+40)
            self.hu = (0.12, 0.35)
            # self.zoom_factor=3

            # self.bbox = (80, 30+40, 150, 100+40)
            # self.zoom_factor=1.5

def test():

    file_name = os.path.basename(__file__)
    filename, _suffix = os.path.splitext(file_name)
    plotter = VisdomLinePlotter(env_name='test')
    
    namespace = NameSpace(args)
    # if args.dataset_name:
        # data_name = args.supervised_dataset_name
    # else:
    data_name = 'MR_ART'

    conditional_filename =  './results/{}/{}/'.format(filename, args.shift) 
    args.PREFIX = conditional_filename
    mkdir(conditional_filename)


    _traindataset, val_loader = get_tta_ct_dataset(args.supervised_dataset_name, test=True)
    # _traindataset, val_loader = get_h5py_unsupervised_dataset(args.unsupervised_dataset_name, test=True, slight_motion=False)

    dataloader = val_loader
    
    img_size = (256, 256)
    # sample_net = Hand_Tailed_Mask_Layer(args.acc, 'cartesian', img_size, resize_size=img_size).to(device, dtype=torch.float)
    physics = CT(256, args.n_views, circle=False, device=device)

    # model MARC
    modelset = EI(args, test=True)
    ei_model = modelset.model  
    ei_model.eval()

    # args.supervised_dataset_name = 'ixi_t1_periodic_slight'
    args.stage = '1'
    modelset = Our(args, test=True)
    our_model = modelset.model
    our_model.eval()

    # model BSA
    # args.supervised_dataset_name = args.pretrain_dataset
    args.stage = '1'
    modelset = TTT(args, test=True)
    ttt_model = modelset.model     
    ttt_model.eval()

    args.stage = '1'
    modelset = Metainvnet(args, test=True)
    metainvnet_model = modelset.model     
    metainvnet_model.net.eval()

    # model cycleGAN med v2
    # args.supervised_dataset_name = args.pretrain_dataset
    # temp_ACC2X = False
    # if args.n_views == 120:
    #     args.n_views = 60
    #     temp_ACC2X = True
    args.stage = '1'
    modelset = LEARN(args, test=True)
    mdr_model = modelset.model  
    mdr_model.eval()

    # if temp_ACC2X:
    #     args.n_views = 120



    imgs = {}
    masks = {}
    segs = {}
    # df = {'Corrupted': None, 'MARC': None, 'BSA': None, 'OursPrior': None, 'Ours':None}
    df = {'ZF': None, 'EI': None, 'LEARN': None, 'TTT': None, 'MeteInvNet': None, 'Ours':None}

    template = dataframe_template(dataset_name=data_name)
    # initialize
    for key in df.keys():
        df[key] = pd.DataFrame(template)
    keys_list = list(df.keys())
    keys_list.insert(0, 'GT')
    NUM_SLICE = 0
    # patient_namelists = data_gallary[data_name](test=isTestdata,  args=args)[0] # 把每个slice的名称存起来。好消息：每个文件名自带患者信息。
    with torch.no_grad():
        for x_iter in dataloader:
            still, with_motion, _shotname = x_iter
            inputs = still.to(device, dtype=torch.float)

            still = still.to(device, dtype=torch.float)

            meas0 = physics.A(still)
            # s_mpg = torch.log(physics.I0 / meas0)
            fbp_mpg = physics.A_dagger(meas0)

            shotname = _shotname[0]  # dataloader makes string '1' to list ('1',)

            imgs['GT'] = still.detach().cpu().numpy()[0, 0, ...]
            imgs['ZF'] = fbp_mpg.detach().cpu().numpy()[0, 0, ...]


            pred_results = ei_model(fbp_mpg, meas0)
            imgs['EI'] = pred_results.detach().cpu().numpy()[0, 0, ...]

            # pred_results = mdr_model(fbp_mpg, meas0)
            imgs['LEARN'] = pred_results.detach().cpu().numpy()[0, 0, ...]

            # pred_results = ttt_model(fbp_mpg, meas0)
            imgs['TTT'] =  pred_results.detach().cpu().numpy()[0, 0, ...]

            output = metainvnet_model.tr_model(meas0, fbp_mpg)
            pred_results = output[-1]
            imgs['MeteInvNet'] =  pred_results.detach().cpu().numpy()[0, 0, ...]

            # pred_results = our_model(fbp_mpg, meas0)
            imgs['Ours']  = pred_results.detach().cpu().numpy()[0, 0, ...]

            get_metrics(args, imgs, df, shotname, NUM_SLICE)  # ? update psnrs and ssims , and get the current metrics - range -1~2000
            # get_ct_map(args, imgs, NUM_SLICE, df, data_name, shotname, keys_list, namespace, dpi=300, twoline=True) # for one line.


            for dfkey in df.keys():
                if dfkey in ['shotname', 'slice']:
                    continue
                else:
                    print('{}:'.format(dfkey))
                    print(df[dfkey].mean())
            NUM_SLICE += 1
            # break
        
        excel_filename_pre = conditional_filename + time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
        excel_filename = excel_filename_pre + '.xlsx' 

        newdf2excel(excel_filename, df)
        print('Saving {}'.format(excel_filename))
        # toolbox.read_excel_and_plot(excel_filename, excel_filename_pre, 'png')
#-----------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", type=int, default=1, choices=[0,1], help="The learning rate") # 
    parser.add_argument("--lr", type=float, default=5e-4, help="The learning rate") # 5e-4, 5e-5, 5e-6
    parser.add_argument(
        "--batch_size", type=int, default=12, help="The batch size")
    parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        choices=[0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.20, 0.25],  # cartesian_0.1 is bad;
        help="The undersampling rate")
   
    parser.add_argument(
        "--mask",
        type=str,
        default="random",
        choices=["cartesian", "radial", "random"],
        help="The type of mask")
              
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="The GPU device")
    parser.add_argument(
        "--bn",
        type=bool,
        default=False,
        choices=[True, False],
        help="Is there the batchnormalize")
    parser.add_argument(
        "--model",
        type=str,
        default="model",
        help="which model")
    parser.add_argument(
        "--supervised_dataset_name",
        type=str,
        default='ixi_t1_periodic_slight',
        help="which dataset",
        # choices=['IXI', 'fastMRI']
        )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default='MR_ART',
        help="which dataset",
        # choices=['MR_ART']
        )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='ixi_t1_periodic_slight',
        # choices=['ACDC', 'BrainTS', 'OAI', 'MRB'],
        # choices=['IXI', 'fastMRI', 'MR_ART'],
        help="which dataset")
    parser.add_argument(
        "--test",
        type=bool,
        default=True,
        choices=[True],
        help="If this program is a test program.")
    parser.add_argument(
        "--mask_type",
        type=str,
        default="2D",
        choices=['1D', '2D'])
    parser.add_argument(
        "--nclasses",
        type=int,
        default=None,
        choices=[0,1,2,3,4,5,6,7,8],
        help="nclasses of segmentation")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--PREFIX",
        type=str,
        default="./testing/")
    parser.add_argument(
        "--stage",
        type=str,
        default='3',
        choices=['1','2', '3'],
        help="1: coarse rec net; 2: fixed coarse then train seg; 3: fixed coarse and seg, then train fine; 4: all finetune.")
    parser.add_argument(
        "--stage1_epochs", type=int, default=800, help="The weighted parameter") #
    parser.add_argument(
        "--stage2_epochs", type=int, default=800, help="The weighted parameter") #
    parser.add_argument(
        "--stage3_epochs", type=int, default=800, help="The weighted parameter") #
    parser.add_argument(
        "--load_model",
        type=int,
        default=1,
        choices=[0,1],
        help="reload last parameter?")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None)
    parser.add_argument(
        "--maskType",
        type=str,
        default='2D',
        choices=['1D', '2D'])
    parser.add_argument(
        "--direction",
        type=str,
        default='Axial',
        choices=['Axial', 'Coronal', 'Sagittal'])
    parser.add_argument(
        "--headmotion",
        type=str,
        default=None,
        choices=['Slight', 'Heavy'])
    parser.add_argument(
        "--acc",
        type=int,
        default=4)

    parser.add_argument(
        "--pretrain_dataset",
        type=str,
        default=None)
    parser.add_argument(
        "--shift",
        type=str,
        default=None)
    parser.add_argument(
        "--n_views",
        type=int,
        default=60)
    parser.add_argument(
        "--real_n_views",
        type=int,
        default=60)
    
    parser.add_argument('--log', default=False, help='write output to file')
    parser.add_argument('--phase', type=str, default='tr', help='tr')
    parser.add_argument('--gpu_idx', type=int, default=0, help='idx of gpu')
    parser.add_argument('--data_type', default='dcm', help='dcm, png')
    # Training Parameters
    parser.add_argument('--tr_batch', type=int, default=1, help='batch size')
    parser.add_argument('--layers', type=int, default=3, help='net layers')
    parser.add_argument('--deep', type=int, default=17, help='depth')
    parser.add_argument('--img_size', default=[512,512], help='image size')
    parser.add_argument('--sino_size', nargs='*', default=[360,600], help='sino size')
    parser.add_argument('--poiss_level',default=1e5, help='Poisson noise level')
    parser.add_argument('--gauss_level',default=[0.05], help='Gaussian noise level')

    args = parser.parse_args()
    torch.cuda.set_device(args.cuda)
    test()


    # TODO: For seg only. should be writen in to `test(arg_data=dataset)`
    # TODO: {DATASET_TRAJEC}, MRB_Radial
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
from data.dataset import get_tta_noise_dataset as get_tta_dataset

from utils.train_utils import mkdir
from utils.visdom_visualizer import VisdomLinePlotter
from testing.toolbox.test_toolbox import tensor2np, get_metrics, get_error_map, newdf2excel, dataframe_template, get_seg_oasis1_main_map, get_seg_ablation_map

# load models
from model_backbones.recon_net import Hand_Tailed_Mask_Layer

from training.mri_py_noise.noise_ei_tta import ModelSet as EI
from training.mri_py.main_mdrecon import ModelSet as MDRNet
from training.mri_py_noise.noise_ttt_tta import ModelSet as TTT
# from training.mri_py.main_ttt_train import ModelSet as TTT

# from training.mri_py.our_s1_train import ModelSet as Our
from training.mri_py_noise.noise_our_s2_tta import ModelSet as Our


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

        # if figs == 2:
        #     self.bbox1 = (128, 100, 128+30, 100+30)
        #     self.bbox2 = (95, 170, 95+30, 170+30) 
        #     self.THE_ONE = 1 # 1106

        # elif figs == 3:
        #     # self.bbox1 = (90, 70, 90+60, 70+30)
        #     # self.bbox2 = (110, 150, 110+60, 150+30) 
        #     # self.THE_ONE = 1348

        #     # self.bbox1 = (81, 75, 81+80, 75+40)
        #     self.bbox1 = (30, 50, 30+120, 50+60)
        #     self.bbox2 = (90, 180, 90+80, 180+40) 
        #     self.THE_ONE = 1467
        # x1, y1, x2, y2

        self.zoom_factor=1.5
        if args.shift == "anatomy":
            self.bbox = (100, 100+40, 190, 190+40)
        elif args.shift == "dataset":
            self.bbox = (100, 10+40, 190, 100+40) 
        elif args.shift == "modality":
            self.bbox = (80, 30+40, 150, 100+40) 
            
        elif args.shift == "ratio":
            # self.bbox = (100, 60+40, 140, 100+40)
            # self.zoom_factor=3

            self.bbox = (80, 30+40, 150, 100+40) 
            self.zoom_factor=1.5

def test():

    file_name = os.path.basename(__file__)
    filename, _suffix = os.path.splitext(file_name)
    plotter = VisdomLinePlotter(env_name='test')
    
    namespace = NameSpace(args)
    # if args.dataset_name:
        # data_name = args.supervised_dataset_name
    # else:
    data_name = 'MR_ART'

    conditional_filename =  './results/{}/{}/{}'.format(filename, args.shift, args.sigma) 
    args.PREFIX = conditional_filename
    mkdir(conditional_filename)


    _traindataset, val_loader = get_tta_dataset(args.supervised_dataset_name, sigma=args.sigma, test=True)
    # _traindataset, val_loader = get_h5py_unsupervised_dataset(args.unsupervised_dataset_name, test=True, slight_motion=False)

    dataloader = val_loader
    
    img_size = (256, 256)
    sample_net = Hand_Tailed_Mask_Layer(args.acc, 'cartesian', img_size, resize_size=img_size).to(device, dtype=torch.float)

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



    # model cycleGAN med v2
    # args.supervised_dataset_name = args.pretrain_dataset
    temp_ACC2X = False
    if args.acc == 2:
        args.acc = 4
        temp_ACC2X = True    
    modelset = MDRNet(args, test=True)
    mdr_model = modelset.model  
    mdr_model.eval()

    if temp_ACC2X:
        args.acc = 2

    # mode = ours
    # args.supervised_dataset_name = ourtta
    # args.stage = '1'
    # modelset = OursModeltta(args, test=True)
    # our_model = modelset.model
    # our_model.eval()



    imgs = {}
    masks = {}
    segs = {}
    # df = {'Corrupted': None, 'MARC': None, 'BSA': None, 'OursPrior': None, 'Ours':None}
    df = {'ZF': None, 'EI': None, 'MDReconNet': None, 'TTT': None, 'Ours':None}

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
            inputs = with_motion.to(device, dtype=torch.float)
            still = still.to(device, dtype=torch.float)

            out = sample_net(inputs)

            # x_train_rec = self.model(out['masked_k'] , out['zero_filling'], out['batch_trajectories'])



            shotname = _shotname[0]  # dataloader makes string '1' to list ('1',)
            # imgs['GT'] = tensor2np(torch.norm(still, dim=1, keepdim=True))
            # imgs['GT'] = tensor2np(still)
            still = torch.norm(still, dim=1, keepdim=True)
            imgs['GT'] = still.detach().cpu().numpy()[0, 0, ...]

            # imgs['ZF'] = tensor2np(torch.norm(out['zero_filling'], dim=1, keepdim=True))
            imgs['ZF'] = torch.norm(out['zero_filling'], dim=1, keepdim=True).detach().cpu().numpy()[0, 0, ...]

            pred_results = ei_model(out['masked_k'] , out['zero_filling'], out['batch_trajectories'])
            pred_results = torch.norm(pred_results, dim=1, keepdim=True)
            # imgs['EI'] = tensor2np(pred_results)
            imgs['EI'] = pred_results.detach().cpu().numpy()[0, 0, ...]

            pred_results = mdr_model(out['masked_k'] , out['zero_filling'], out['batch_trajectories'])
            pred_results = torch.norm(pred_results, dim=1, keepdim=True)
            imgs['MDReconNet'] = pred_results.detach().cpu().numpy()[0, 0, ...]

            pred_results = ttt_model(out['masked_k'] , out['zero_filling'], out['batch_trajectories'])
            pred_results = torch.norm(pred_results, dim=1, keepdim=True)
            imgs['TTT'] =  pred_results.detach().cpu().numpy()[0, 0, ...]

            # pred_results = cycle_model(inputs)
            # imgs['OursPrior'] = tensor2np(pred_results)

            pred_results = our_model(out['masked_k'] , out['zero_filling'], out['batch_trajectories'])
            pred_results = torch.norm(pred_results, dim=1, keepdim=True)

            imgs['Ours']  = pred_results.detach().cpu().numpy()[0, 0, ...]

            get_metrics(args, imgs, df, shotname, NUM_SLICE)  # ? update psnrs and ssims , and get the current metrics - range -1~2000
            # get_seg_ablation_map(args, imgs, NUM_SLICE, df, data_name, shotname, keys_list, namespace, dpi=200, oneline=True) # for one line.
            get_seg_ablation_map(args, imgs, NUM_SLICE, df, data_name, shotname, keys_list, namespace, dpi=300, twoline=True) # for one line.

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
        "--sigma",
        type=float,
        default=None)

    args = parser.parse_args()
    torch.cuda.set_device(args.cuda)
    test()


    # TODO: For seg only. should be writen in to `test(arg_data=dataset)`
    # TODO: {DATASET_TRAJEC}, MRB_Radial
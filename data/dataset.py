import sys
import warnings
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

sys.path.append('.') # 

from data.build_h5py_data import H5PYMixedSliceData, H5PY_Supervised_Data, H5PY_Unsupervised_Data, DIV2KDataset, TTA_MRIDataset, TTA_MRIDataset_Noise
from data.build_ct_dataset  import CT_dataset

from data.build_h5py_data_threeslices import H5PY_Supervised_SlicesData, H5PY_Unsupervised_SlicesData
from utils.visdom_visualizer import VisdomLinePlotter


warnings.filterwarnings("ignore")

def get_h5py_mixed_dataset(dataset_name):
    train_dataset=H5PYMixedSliceData(dataset_name)
    val_loader = DataLoader(H5PYMixedSliceData(dataset_name, validation=True), batch_size=1, shuffle=False)
    return train_dataset, val_loader

def get_h5py_mixed_test_dataset(dataset_name):
    val_loader = DataLoader(H5PYMixedSliceData(dataset_name, test=True), batch_size=1, shuffle=False)
    return val_loader

def get_h5py_supervised_dataset(dataset_name, tta_testing=False, bezier=False, test=False):
    train_dataset=H5PY_Supervised_Data(dataset_name, tta_testing=tta_testing, bezier=bezier)
    val_loader = DataLoader(H5PY_Supervised_Data(dataset_name, validation=True, test=test), batch_size=1, shuffle=False)
    return train_dataset, val_loader

def get_h5py_unsupervised_dataset(dataset_name, test=False, slight_motion=False, one_shot=False):
    train_dataset=H5PY_Unsupervised_Data(dataset_name, one_shot=one_shot)
    val_loader = DataLoader(H5PY_Unsupervised_Data(dataset_name, validation=True, test=test, slight_motion=slight_motion), batch_size=1, shuffle=False)
    return train_dataset, val_loader

def get_h5py_supervisedSlices_dataset(dataset_name):
    train_dataset=H5PY_Supervised_SlicesData(dataset_name)
    val_loader = DataLoader(H5PY_Supervised_SlicesData(dataset_name, validation=True), batch_size=1, shuffle=False)
    return train_dataset, val_loader

def get_h5py_unsupervisedSlices_dataset(dataset_name):
    train_dataset=H5PY_Unsupervised_SlicesData(dataset_name)
    val_loader = DataLoader(H5PY_Unsupervised_SlicesData(dataset_name, validation=True), batch_size=1, shuffle=False)
    return train_dataset, val_loader


# for compared methods
def get_h5py_supervised_dataset_comparedmethods(dataset_name):
    train_dataset=H5PY_Supervised_Data(dataset_name, change_isotopic=False)
    val_loader = DataLoader(H5PY_Supervised_Data(dataset_name, change_isotopic=False, validation=True), batch_size=1, shuffle=False)
    return train_dataset, val_loader


def get_h5py_unsupervised_dataset_comparedmethods(dataset_name):
    train_dataset=H5PY_Unsupervised_Data(dataset_name, change_isotopic=False)
    val_loader = DataLoader(H5PY_Unsupervised_Data(dataset_name, change_isotopic=False, validation=True), batch_size=1, shuffle=False)
    return train_dataset, val_loader

def get_tta_dataset(dataset_name, test=False, one_shot=False):
    train_dataset=TTA_MRIDataset(dataset_name, one_shot=one_shot)
    val_loader = DataLoader(TTA_MRIDataset(dataset_name, validation=True, test=test), batch_size=1, shuffle=False)
    return train_dataset, val_loader

def get_tta_noise_dataset(dataset_name, sigma, test=False, one_shot=False):
    train_dataset=TTA_MRIDataset_Noise(dataset_name, sigma, one_shot=one_shot)
    val_loader = DataLoader(TTA_MRIDataset_Noise(dataset_name, sigma, validation=True, test=test), batch_size=1, shuffle=False)
    return train_dataset, val_loader

def get_tta_ct_dataset(dataset_name, test=False, one_shot=False):
    train_dataset=CT_dataset(dataset_name, one_shot=one_shot)
    val_loader = DataLoader(CT_dataset(dataset_name, validation=True, test=test), batch_size=1, shuffle=False)
    return train_dataset, val_loader

if __name__ == "__main__":

    file_name = os.path.basename(__file__)
    plotter = VisdomLinePlotter(env_name=file_name)

    dataset_sup, val_loader_sup = get_h5py_supervised_dataset('ixi_t1_periodic_slight_sagittal', bezier=True)
    # dataset_sup, val_loader_sup = get_h5py_supervised_dataset('ixi_t1_periodic_moderate_sagittal', bezier=False)

    dataloader_sup = DataLoader(dataset_sup, 
                            batch_size=1, 
                            shuffle=False,
                            num_workers=4)  # 增加GPU利用率稳定性。
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch=0
    loop = tqdm(enumerate(val_loader_sup), total=len(val_loader_sup))
    for index, (still, with_motion, _shotname) in loop:

        x_train = still.to(device, dtype=torch.float)
        y_train = with_motion.to(device, dtype=torch.float)
        plotter.image('still_sup', epoch, x_train)
        plotter.image('with_motion', epoch, y_train)
        epoch+=1

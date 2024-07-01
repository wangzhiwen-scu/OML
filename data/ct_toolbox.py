import os
import numpy as np
import cv2
import torch
import glob
import h5py
import sys
import SimpleITK as sitk

import imageio.v3 as iio

import random

def get_random_elements(data, m, seed_value):
    random.seed(seed_value)
    return random.sample(data, m)



sys.path.append('.') 

def crop_arr(arr, size):
    """crop img_arr in dataloader. before transform.; CENTERCROP
        arr = (h, w), size is target size.
    """
    h, w = arr.shape[0], arr.shape[1]
    th, tw = size[0], size[1]
    crop_img = arr[int(h/2)-int(th/2):int(h/2)+int(th/2), int(w/2)-int(tw/2):int(w/2)+int(tw/2)]
    return crop_img

def resize_vol_img(arr, size):
    """crop img_arr in dataloader. before transform.; CENTERCROP
        arr = (slice, h, w), size is target size.
    """
    s, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
    new_arr = np.zeros((s, size, size))
    for i in range(arr.shape[0]):
        new_arr[i] = cv2.resize(arr[i], (size, size), interpolation=cv2.INTER_LINEAR)
    return new_arr

def resize_vol_seg(arr, size):
    """resize img_arr in dataloader. before transform.; CENTERCROP
        arr = (slice, channel,h, w), size is target size.
    """
    s, c, h, w = arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3]
    new_arr = np.zeros((s, c, size, size))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i][j] = cv2.resize(arr[i][j], (size, size), interpolation=cv2.INTER_LINEAR)
    return new_arr

def get_filePath_fileName_fileExt(fileUrl):
    filepath, tmpfilename = os.path.split(fileUrl)
    shotname, extension = os.path.splitext(tmpfilename)
    return filepath, shotname, extension

def get_mrb_shotname(fileUrl):
    filepath, _shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    if 'MRBrainS13DataNii' in filepath: 
        realshotname = filepath.split("/")[-1]
    elif '18training_corrected' in filepath:
        realshotname = filepath.split("/")[-2]
    
    return realshotname

def get_acdc_shotname(fileUrl):
    filepath, _shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    realshotname = filepath.split("/")[-1]

    
    return realshotname

def get_braints_shotname(fileUrl):
    filepath, _shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    realshotname = filepath.split("/")[-1]

    print(realshotname)
    return realshotname

def get_oai_shotname(fileUrl):
    filepath, shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    

    print(shotname)
    return shotname

def get_miccai_shotname(fileUrl):
    filepath, shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    

    
    return shotname

def get_h5py_shotname(fileUrl):
    filepath, shotname, _extension = get_filePath_fileName_fileExt(fileUrl)

    # print(shotname)
    return shotname

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def torch0to1(img):
    B, C, H, W = img.shape
    for b in range(B):
        img_min = torch.min(img[b, :, :,:])
        img_max = torch.max(img[b, :, :,:])
        img[b, :, :,:] = 1.0 * (img[b, :, :,:] - img_min) / (img_max - img_min)
    return img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()



class CT_path(object):

    @staticmethod
    def get_mayo_path_lists():
        raw_path = r'./data/datasets/TTA_datasets/CT/Mayo/'
        train_img_path = glob.glob(raw_path+'/train/*/full_3mm/*.IMA') # read by dicom
        test_img_path = glob.glob(raw_path+'/test/*/full_3mm/*.IMA') # read by dicom

        # Sample list


        train_img_path = get_random_elements(train_img_path, m=100, seed_value=42)
        test_img_path = get_random_elements(test_img_path, m=100, seed_value=42)


        # train_img_path = train_img_path[0:100]
        # train_img_path = train_img_path[0:30]

        return train_img_path, test_img_path
    @staticmethod
    def get_mayo_dicom(fname, slice):
        item_img = sitk.GetArrayFromImage(sitk.ReadImage(fname)) # (slice, h,w)
        return item_img
        


    @staticmethod
    def get_deeplesion_path_lists():
        raw_path = r'./data/datasets/TTA_datasets/CT/WT/deep lesion/'
        train_img_path = glob.glob(raw_path+'*/*.png')
        test_img_path = glob.glob(raw_path+'*/*.png')

        # train_img_path = train_img_path[0:100]

        train_img_path = get_random_elements(train_img_path, m=100, seed_value=42)
        test_img_path = get_random_elements(test_img_path, m=100, seed_value=42)

        return train_img_path, test_img_path
    @staticmethod
    def get_deeplesion_dicom(fname, slice):
        item_img = iio.imread(fname)
        return item_img
    
    @staticmethod
    def get_spine_path_lists():
        raw_path = r'./data/datasets/TTA_datasets/CT/WT/spine/spine-1/'
        all_path = glob.glob(raw_path+'*/*/*.nii.gz')

        train_img_path = all_path[0:10]
        test_img_path = all_path[10:20]
        return train_img_path, test_img_path
    
    @staticmethod
    def get_spine_dicom(fname, slice):
        item_img = sitk.GetArrayFromImage(sitk.ReadImage(fname))[slice, :,:] # (slice, h,w)
        return item_img
    
    @staticmethod
    def get_cta_path_lists():
        raw_path = r'./data/datasets/TTA_datasets/CT/Eudora/CTA_huaxi/'
        train_img_path = glob.glob(raw_path+'*.nii.gz')
        train_img_path = train_img_path[0:10]
        test_img_path = glob.glob(raw_path+'*.nii.gz')
        test_img_path = test_img_path[10:20]
        return train_img_path, test_img_path
    @staticmethod
    def get_cta_dicom(fname, slice):
        item_img = sitk.GetArrayFromImage(sitk.ReadImage(fname))[slice, :,:] # (slice, h,w)
        return item_img
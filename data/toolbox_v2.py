import os
import SimpleITK as sitk
import numpy as np
import cv2
import torch
import glob
import h5py

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
    """获取文件路径, 文件名, 后缀名
    :param fileUrl:
    :return
    """
    filepath, tmpfilename = os.path.split(fileUrl)
    shotname, extension = os.path.splitext(tmpfilename)
    return filepath, shotname, extension

def get_mrb_shotname(fileUrl):
    filepath, _shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    if 'MRBrainS13DataNii' in filepath: 
        realshotname = filepath.split("/")[-1]
    elif '18training_corrected' in filepath:
        realshotname = filepath.split("/")[-2]
    # print(realshotname)
    return realshotname

def get_acdc_shotname(fileUrl):
    filepath, _shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    realshotname = filepath.split("/")[-1]

    # print(realshotname)
    return realshotname

def get_braints_shotname(fileUrl):
    filepath, _shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    realshotname = filepath.split("/")[-1]

    print(realshotname)
    return realshotname

def get_oai_shotname(fileUrl):
    filepath, shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    # realshotname = filepath.split("/")[-1]

    print(shotname)
    return shotname

def get_miccai_shotname(fileUrl):
    filepath, shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    # realshotname = filepath.split("/")[-1]

    # print(shotname)
    return shotname

def get_h5py_shotname(fileUrl):
    filepath, shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    # realshotname = filepath.split("/")[-1]

    print(shotname)
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



class MRB(object):
    @staticmethod
    def get_MRBrains18_raw(choice=['1', '4', '5', '7', '14', '070', '148']):
        # raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/MRBrainS13DataNii/TrainingData/'
        raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/18training_corrected/training/'

        subdir = [os.path.join(raw_path, f) for f in os.listdir(raw_path)]
        T1img = [os.path.join(f, 'pre/reg_T1.nii.gz') for f in subdir]
        # Label = [os.path.join(f, 'LabelsForTraining.nii') for f in subdir] # training raw segmentation
        Label = [os.path.join(f, 'segm.nii.gz') for f in subdir] # test specific segmentation
        # raw_y = r'/home/ubuntu/wzw/COMBINE_PYTORCH/data/OAI/train/*.seg'

        all_paths = [(T1img[i], Label[i]) for i in range(len(T1img))]

        # T1img, Label = np.array(T1img), np.array(Label)
        # T1img, Label = np.sort(T1img, axis=0), np.sort(Label, axis=0)
        # img 和 seglable 组成一队。
        
        # x_abspath = [f for f in glob.glob(raw_x)]
        # y_abspath = [f for f in glob.glob(raw_y)]
        # train_indices = [0, 2, 4, 6] # 5, 070, 4, 148, 
        train_indices = [2, 3] # 070, 148, 
        train_data = [all_paths[index] for index in train_indices]

        # test_indices = [1, 3, 5]  # 14, 7, 1
        test_indices = [1, 6]  # 14, 7
        test_data = [all_paths[index] for index in test_indices]

        return train_data, test_data

    @staticmethod
    def get_MRBrains13_raw(choice=['1', '2', '3', '4', '5']):
        raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/MRBrainS13DataNii/TrainingData/'
        subdir = [os.path.join(raw_path, f) for f in os.listdir(raw_path)]
        T1img = [os.path.join(f, 'T1.nii') for f in subdir]
        Label = [os.path.join(f, 'LabelsForTraining.nii') for f in subdir] # training raw segmentation

        all_paths = [(T1img[i], Label[i]) for i in range(len(T1img))]

        # x_abspath = [f for f in glob.glob(raw_x)]
        # y_abspath = [f for f in glob.glob(raw_y)]
        train_indices = [0, 1]
        train_data = [all_paths[index] for index in train_indices]
        test_indices = [2, 3, 4]
        test_data = [all_paths[index] for index in test_indices]
        return train_data, test_data

    @staticmethod
    def merge13and18():
        train_data18, test_data18 = MRB.get_MRBrains18_raw()
        train_data13, test_data13 = MRB.get_MRBrains13_raw()
        train_data = train_data18 + train_data13
        test_data = test_data18 + test_data13
        return train_data, test_data

    @staticmethod
    def getslicefromfile(fname, slice, imgorseg):
        if imgorseg == 'img':
            index = 0
        elif imgorseg == 'seg':
            index = 1
        # with sitk.GetArrayFromImage(sitk.ReadImage(fname[index])) as data: # ((img_file, seg_file), shotname, slice)/
        data = sitk.GetArrayFromImage(sitk.ReadImage(fname[index]))
        # kspace = data['kspace'][slice]
        # mask = np.asarray(data['mask']) if 'mask' in data else None
        img_ = data[slice]
        # img_ = cv2.resize(img_, (240, 240))
        
        # img_=np.rot90(img_)
        # img_ = cv2.flip(img_,1)
        
        img_ = img_.astype(np.double)
        # attrs = dict(data.attrs)
        # attrs['padding_left'] = padding_left
        # attrs['padding_right'] = padding_right
        # return kspace, mask, target, attrs, fname.name, slice
        # return x_train, shotname, slice
        return img_

    @staticmethod
    def mrb2onehot(mrb_seg):
        mrb_seg = mrb_seg.astype(np.uint8)
        mrb_seg[mrb_seg >=9 ] = 3 #  9 and 10, i.e. infarctions and ‘other’ lesions,  you may label these voxels as gray matter, white matter, or any other label. 
        mrb_seg[mrb_seg >= 4] = mrb_seg[mrb_seg >= 4] - 1 # all label minus one, because of 4 is white matter lesions.

        nclasses = 7 # MRB for training lable
        var_nclasses = nclasses + 1
        # one_hot_lbl = np.zeros((nclasses+1, 240, 240))
        one_hot_lbl = mask2onehot(mrb_seg, nclasses+1)
        one_hot_lbl = one_hot_lbl[1:var_nclasses, ...]
        return one_hot_lbl

    @staticmethod
    def derivative(onehotseglabels):
        '''https://blog.csdn.net/lovetobelove/article/details/86618324
        input: range(0,1)
        '''
        img = onehotseglabels
        i = 0
        img = img*255 # range(0,1)
        result = np.zeros_like(img)
        for i in range(img.shape[0]):
            # result[i]= cv2.Canny(img[i], 50, 150) 
            result[i]=cv2.Laplacian(img[i],cv2.CV_64F)
            result[i] = cv2.convertScaleAbs(result[i])  #绝对值转换
        output = result.copy()
        output[result< 255/2] = 0
        output[result> 255/2] = 1
        # output=np.rot90(output)
        # output = cv2.flip(output,1)
        return output

class OAI(object):
    @staticmethod
    def get_all_oai_root(choice=['1', '2', '3', '4', '5']):
        train_path = r'./datasets/datasets/OAI/train/'
        train_im_path = glob.glob(train_path + '*.im')
        # train_seg_path = glob.glob(train_path + '*.seg')
        train_seg_path = [f[0:-3]+'.seg' for f in train_im_path]
        train_all_paths = [(train_im_path[i], train_seg_path[i]) for i in range(len(train_im_path))]
        # train_data = train_all_paths[0:10]
        train_data = train_all_paths

        test_path = r'./datasets/datasets/OAI//valid/'
        test_im_path = glob.glob(test_path + '*.im')
        test_seg_path = [f[0:-3]+'.seg' for f in test_im_path]
        test_all_paths = [(test_im_path[i], test_seg_path[i]) for i in range(len(test_im_path))]
        test_data = test_all_paths[0:5]

        return train_data, test_data

    @staticmethod
    def get_oai_h5py():
        # raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/MRBrainS13DataNii/TrainingData/'
        # https://blog.csdn.net/weixin_40313940/article/details/105799626 标签不对齐。
        # raw_path = r'/home/harddisk8T/wzw/datasets/MICCAI2012/release/MICCAI-2012-Multi-Atlas-Challenge-Data/' # harddist8T
        # raw_path = r'/home/labuser1/wzw/small_datasets/OAI/OAI-h5py-old/'
        raw_path = r'./datasets/datasets/OAI/'

        train_img_path = glob.glob(raw_path+'training-h5py/*.h5')
        test_img_path = glob.glob(raw_path+'validation-h5py/*.h5')

        return train_img_path, test_img_path

    @staticmethod
    def get_oai_edge_h5py():
        # raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/MRBrainS13DataNii/TrainingData/'
        # https://blog.csdn.net/weixin_40313940/article/details/105799626 标签不对齐。
        # raw_path = r'/home/harddisk8T/wzw/datasets/MICCAI2012/release/MICCAI-2012-Multi-Atlas-Challenge-Data/' # harddist8T
        raw_path = r'./datasets/datasets/OAI//'

        train_img_path = glob.glob(raw_path+'training-h5py/*.h5')
        test_img_path = glob.glob(raw_path+'validation-h5py/*.h5')

        return train_img_path, test_img_path

    @staticmethod
    def getslicefromfile(fname, slice, imgorseg):
        if imgorseg == 'img':
            index = 0
        elif imgorseg == 'seg':
            index = 1  # (384, 384, 160, 6)
        with h5py.File(fname[index], 'r') as data:
            data = data['data']
            img_ = data[:,:, slice]
            img_ = img_.astype(np.double)
            return img_

    @staticmethod
    def oai2onehot(mrb_seg):
        mrb_seg = mrb_seg.astype(np.uint8)
        # mrb_seg[mrb_seg >=9 ] = 3 #  9 and 10, i.e. infarctions and ‘other’ lesions,  you may label these voxels as gray matter, white matter, or any other label. 
        # mrb_seg[mrb_seg >= 4] = mrb_seg[mrb_seg >= 4] - 1 # all label minus one, because of 4 is white matter lesions.

        nclasses = 6 # oai for training; lable OAI has 6 lables.
        var_nclasses = nclasses + 1
        # one_hot_lbl = np.zeros((nclasses+1, 240, 240))
        one_hot_lbl = mask2onehot(mrb_seg, nclasses+1)
        one_hot_lbl = one_hot_lbl[1:var_nclasses, ...]
        return one_hot_lbl

    @staticmethod
    def derivative(onehotseglabels):
        '''https://blog.csdn.net/lovetobelove/article/details/86618324
        input: range(0,1)
        '''
        img = onehotseglabels
        i = 0
        img = img*255 # range(0,1)
        result = np.zeros_like(img)
        for i in range(img.shape[0]):
            # result[i]= cv2.Canny(img[i], 50, 150) 
            result[i]=cv2.Laplacian(img[i],cv2.CV_64F)
            result[i] = cv2.convertScaleAbs(result[i])  #绝对值转换
        output = result.copy()
        output[result< 255/2] = 0
        output[result> 255/2] = 1
        return output


class ACDC(object):

    @staticmethod
    def get_all_acdc_root(choice=['1', '2', '3', '4', '5']):
        #Info.cfg  patient001_4d.nii.gz  patient001_frame01_gt.nii.gz  patient001_frame01.nii.gz  patient001_frame12_gt.nii.gz  patient001_frame12.nii.gz
        raw_path = r'/home/harddisk8T/wzw/datasets/ACDC_cineMRI/training/'   # 100 pantients.
        all_img_path = glob.glob(raw_path+'*/*frame01.nii.gz')
        all_seg_path = [f[0:-7]+'_gt.nii.gz' for f in all_img_path]

        all_paths = [(all_img_path[i], all_seg_path[i]) for i in range(len(all_img_path))]

        train_data = [all_paths[index] for index in range(len(all_paths)-29)]

        test_data = [all_paths[index] for index in range(len(all_paths)-29, len(all_paths))]
        return train_data, test_data
    @staticmethod
    def get_acdc_h5py():
        # raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/MRBrainS13DataNii/TrainingData/'
        # https://blog.csdn.net/weixin_40313940/article/details/105799626 标签不对齐。
        # raw_path = r'/home/harddisk8T/wzw/datasets/MICCAI2012/release/MICCAI-2012-Multi-Atlas-Challenge-Data/' # harddist8T
        raw_path = r'/home/labuser1/wzw/small_datasets/ACDC_cineMRI/ACDC-h5py/'
        # train_img_path = glob.glob(raw_path+'training-h5py/*.h5')
        # test_img_path = glob.glob(raw_path+'testing-h5py/*.h5')
        train_img_path = glob.glob(raw_path+'training-h5py/*01.nii.h5')
        # test_img_path = glob.glob(raw_path+'testing-h5py/*1[0-9].nii.h5')
        test_img_path = glob.glob(raw_path+'testing-h5py/*01.nii.h5')

        return train_img_path, test_img_path

    @staticmethod
    def get_acdc_edge_h5py():
        # raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/MRBrainS13DataNii/TrainingData/'
        # https://blog.csdn.net/weixin_40313940/article/details/105799626 标签不对齐。
        # raw_path = r'/home/harddisk8T/wzw/datasets/MICCAI2012/release/MICCAI-2012-Multi-Atlas-Challenge-Data/' # harddist8T
        raw_path = r'/home/labuser1/wzw/small_datasets/ACDC_cineMRI/ACDC-imgedge/'
        # train_img_path = glob.glob(raw_path+'training-h5py/*.h5')
        # test_img_path = glob.glob(raw_path+'testing-h5py/*.h5')
        train_img_path = glob.glob(raw_path+'training-h5py/*01.nii.h5')
        # test_img_path = glob.glob(raw_path+'testing-h5py/*1[0-9].nii.h5')
        test_img_path = glob.glob(raw_path+'testing-h5py/*01.nii.h5')

        return train_img_path, test_img_path
        
    @staticmethod
    def getslicefromfile(fname, slice, imgorseg):
        if imgorseg == 'img':
            index = 0
        elif imgorseg == 'seg':
            index = 1
        data = sitk.GetArrayFromImage(sitk.ReadImage(fname[index]))
        img_ = data[slice] # 8, 256, 216 -> F, S, H, W (分割仅第一帧可用。)
        img_ = img_.astype(np.double)
        return img_

    @staticmethod
    def mrb2onehot(mrb_seg):
        mrb_seg = mrb_seg.astype(np.uint8)
        # mrb_seg[mrb_seg >=9 ] = 3 #  9 and 10, i.e. infarctions and ‘other’ lesions,  you may label these voxels as gray matter, white matter, or any other label. 
        # mrb_seg[mrb_seg >= 4] = mrb_seg[mrb_seg >= 4] - 1 # all label minus one, because of 4 is white matter lesions.

        nclasses = 3 # MRB for training lable
        var_nclasses = nclasses + 1
        # one_hot_lbl = np.zeros((nclasses+1, 240, 240))
        one_hot_lbl = mask2onehot(mrb_seg, nclasses+1)
        one_hot_lbl = one_hot_lbl[1:var_nclasses, ...]
        return one_hot_lbl

    @staticmethod
    def derivative(onehotseglabels):
        '''https://blog.csdn.net/lovetobelove/article/details/86618324
        input: range(0,1)
        '''
        img = onehotseglabels
        i = 0
        img = img*255 # range(0,1)
        result = np.zeros_like(img)
        for i in range(img.shape[0]):
            # result[i]= cv2.Canny(img[i], 50, 150) 
            result[i]=cv2.Laplacian(img[i],cv2.CV_64F)
            result[i] = cv2.convertScaleAbs(result[i])  #绝对值转换
        output = result.copy()
        output[result< 255/2] = 0
        output[result> 255/2] = 1
        return output

    @staticmethod
    def getslicefromh5py(fname, slice):
        with h5py.File(fname, 'r') as data:
            img_ = data['img'][slice]
            seg_ = data['seg'][slice]
            edge = data['edge'][slice]
            img_ = img_.astype(np.double)
            seg_ = seg_.astype(np.double)
            edge = edge.astype(np.double)
            return img_, seg_, edge
            
class BrainTS(object):

    @staticmethod
    def get_all_braints_root(choice=['1', '2', '3', '4', '5']):
        #BraTS19_TCIA06_211_1_flair.nii.gz  BraTS19_TCIA06_211_1_seg.nii.gz  BraTS19_TCIA06_211_1_t1ce.nii.gz  BraTS19_TCIA06_211_1_t1.nii.gz  BraTS19_TCIA06_211_1_t2.nii.gz
        raw_path = r'/home/harddisk8T/wzw/datasets/BrainTS/BrainTS2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/'   # HGG 259 pantients./ LGG: 76
        all_img_path = glob.glob(raw_path+'HGG/*/*t1ce.nii.gz')  # 259, 
        all_seg_path = [f[0:-11]+'seg.nii.gz' for f in all_img_path]

        all_paths = [(all_img_path[i], all_seg_path[i]) for i in range(len(all_img_path))]

        # train_data = [all_paths[index] for index in range(len(all_paths)-99)]
        # test_data = [all_paths[index] for index in range(len(all_paths)-99, len(all_paths))]
        train_data = all_paths[0:50]
        test_data = all_paths[51:60]
        return train_data, test_data

    @staticmethod
    def get_braints_h5py():
        # raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/MRBrainS13DataNii/TrainingData/'
        # https://blog.csdn.net/weixin_40313940/article/details/105799626 标签不对齐。
        # raw_path = r'/home/harddisk8T/wzw/datasets/MICCAI2012/release/MICCAI-2012-Multi-Atlas-Challenge-Data/' # harddist8T
        raw_path = r'/home/labuser1/wzw/small_datasets/BrainTS/BrainTS2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG-h5py-t1ce/'

        train_img_path = glob.glob(raw_path+'training-h5py/*.h5')
        test_img_path = glob.glob(raw_path+'testing-h5py/*.h5')

        return train_img_path, test_img_path
    @staticmethod
    def getslicefromfile(fname, slice, imgorseg):
        if imgorseg == 'img':
            index = 0
        elif imgorseg == 'seg':
            index = 1
        data = sitk.GetArrayFromImage(sitk.ReadImage(fname[index]))
        img_ = data[slice] # 8, 256, 216 -> F, S, H, W (分割仅第一帧可用。)
        img_ = img_.astype(np.double)
        return img_

    @staticmethod
    def mrb2onehot(mrb_seg):
        mrb_seg = mrb_seg.astype(np.uint8)
        # mrb_seg[mrb_seg >=9 ] = 3 #  9 and 10, i.e. infarctions and ‘other’ lesions,  you may label these voxels as gray matter, white matter, or any other label. 
        mrb_seg[mrb_seg ==4] = 3 # all label minus one, because of 4 is white matter lesions.

        nclasses = 3 # MRB for training lable
        var_nclasses = nclasses + 1
        # one_hot_lbl = np.zeros((nclasses+1, 240, 240))
        one_hot_lbl = mask2onehot(mrb_seg, nclasses+1)
        one_hot_lbl = one_hot_lbl[1:var_nclasses, ...]
        return one_hot_lbl

    @staticmethod
    def derivative(onehotseglabels):
        '''https://blog.csdn.net/lovetobelove/article/details/86618324
        input: range(0,1)
        '''
        img = onehotseglabels
        i = 0
        img = img*255 # range(0,1)
        result = np.zeros_like(img)
        for i in range(img.shape[0]):
            # result[i]= cv2.Canny(img[i], 50, 150) 
            result[i]=cv2.Laplacian(img[i],cv2.CV_64F)
            result[i] = cv2.convertScaleAbs(result[i])  #绝对值转换
        output = result.copy()
        output[result< 255/2] = 0
        output[result> 255/2] = 1
        return output

class Miccai(object):
    @staticmethod
    def get_Miccai_raw():
        # raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/MRBrainS13DataNii/TrainingData/'
        # https://blog.csdn.net/weixin_40313940/article/details/105799626 标签不对齐。
        # raw_path = r'/home/harddisk8T/wzw/datasets/MICCAI2012/release/MICCAI-2012-Multi-Atlas-Challenge-Data/' # harddist8T
        raw_path = r'/home/labuser1/wzw/small_datasets/MICCAI2012/release/MICCAI-2012-Multi-Atlas-Challenge-Data/'

        train_img_path = glob.glob(raw_path+'training-images/*.nii.gz')
        train_img_path.sort()
        # train_lable_path = [f[0:-8]+'_glm.nii.gz' for f in train_img_path]
        train_lable_path = glob.glob(raw_path+'training-labels-33/*.nii.gz')
        train_lable_path.sort()
        train_data = [(train_img_path[i], train_lable_path[i]) for i in range(len(train_img_path))]

        test_img_path = glob.glob(raw_path+'testing-images/*.nii.gz')
        test_img_path.sort()
        test_lable_path = glob.glob(raw_path+'testing-labels-33/*.nii.gz')
        test_lable_path.sort()
        test_data = [(test_img_path[i], test_lable_path[i]) for i in range(len(test_img_path))]

        return train_data, test_data[0:1]

    @staticmethod
    def get_Miccai_h5py():
        # raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/MRBrainS13DataNii/TrainingData/'
        # https://blog.csdn.net/weixin_40313940/article/details/105799626 标签不对齐。
        # raw_path = r'/home/harddisk8T/wzw/datasets/MICCAI2012/release/MICCAI-2012-Multi-Atlas-Challenge-Data/' # harddist8T
        raw_path = r'/home/labuser1/wzw/small_datasets/MICCAI2012/release/MICCAI-2012-Multi-Atlas-Challenge-Data/'

        train_img_path = glob.glob(raw_path+'training-h5py/*.h5')
        test_img_path = glob.glob(raw_path+'testing-h5py/*.h5')

        return train_img_path, test_img_path
    @staticmethod
    def getslicefromfile(fname, slice, imgorseg):
        if imgorseg == 'img':
            index = 0
        elif imgorseg == 'seg':
            index = 1
        data = sitk.GetArrayFromImage(sitk.ReadImage(fname[index]))
        img_ = data[slice] # 8, 256, 216 -> F, S, H, W (分割仅第一帧可用。)
        img_ = img_.astype(np.double)
        return img_

    @staticmethod
    def getslicefromh5py(fname, slice):
        with h5py.File(fname, 'r') as data:
            img_ = data['img'][slice]
            seg_ = data['seg'][slice]
            edge = data['edge'][slice]
            img_ = img_.astype(np.double)
            seg_ = seg_.astype(np.double)
            edge = edge.astype(np.double)
            return img_, seg_, edge

    @staticmethod
    def mrb2onehot(mrb_seg):
        mrb_seg = mrb_seg.astype(np.uint8)
        # mrb_seg[mrb_seg >=9 ] = 3 #  9 and 10, i.e. infarctions and ‘other’ lesions,  you may label these voxels as gray matter, white matter, or any other label. 
        # mrb_seg[mrb_seg >= 4] = mrb_seg[mrb_seg >= 4] - 1 # all label minus one, because of 4 is white matter lesions.

        nclasses = 33 # MRB for training lable
        var_nclasses = nclasses + 1
        # one_hot_lbl = np.zeros((nclasses+1, 240, 240))
        one_hot_lbl = mask2onehot(mrb_seg, nclasses+1)
        one_hot_lbl = one_hot_lbl[1:var_nclasses, ...]
        return one_hot_lbl

    @staticmethod
    def vol_mrb2onehot(seg_map_uint):
        """crop img_arr in dataloader. before transform.; CENTERCROP
            seg_vol_one_hot = (slice, channel, h, w), size is target size.
        """
        s, h, w = seg_map_uint.shape[0], seg_map_uint.shape[1], seg_map_uint.shape[2]
        new_arr = np.zeros((s, 33, h, w))
        for i in range(len(seg_map_uint)):
            new_arr[i] = Miccai.mrb2onehot(seg_map_uint[i])
        return new_arr

    @staticmethod
    def derivative(onehotseglabels):
        '''https://blog.csdn.net/lovetobelove/article/details/86618324
        input: range(0,1)
        '''
        img = onehotseglabels
        i = 0
        img = img*255 # range(0,1)
        result = np.zeros_like(img)
        for i in range(img.shape[0]):
            # result[i]= cv2.Canny(img[i], 50, 150) 
            result[i]=cv2.Laplacian(img[i],cv2.CV_64F)
            result[i] = cv2.convertScaleAbs(result[i])  #绝对值转换
        output = result.copy()
        output[result< 255/2] = 0
        output[result> 255/2] = 1
        return output

    @staticmethod
    def getslicefrom_skmtea_imgandseg(fname, slice):

        
        with h5py.File(fname, 'r') as data:
            img_ = data['img'][slice]
            seg_ = data['seg'][slice]
            segedge = data['segedge'][slice]
            imgedge = data['imgedge'][slice]

            img_ = img_.astype(np.double)
            seg_ = seg_.astype(np.double)
            segedge = segedge.astype(np.double)
            imgedge = imgedge.astype(np.double)

            return img_, seg_, segedge, imgedge



    @staticmethod
    def getslicefrom_skmtea_multicoil(shotname, slice, multicoil_root_dir):
        # multicoil_root_dir = r"./datasets/skmteah5/files_recon_calib-24/all-h5py/"
        # multicoil_root_dir = r"./datasets/skmteah5/files_recon_calib-24/training-h5py/"

        image_path = os.path.join(multicoil_root_dir, shotname + '.h5')

        with h5py.File(image_path, 'r') as data:
            kspace = data['kspace'][slice][:,:,0,:]  # [:,:,0:1,:]  echo=0
            maps = data['maps'][slice][:,:,:,0]
            # edge = data['edge'][slice]
            # img_ = img_.astype(np.double)
            # seg_ = seg_.astype(np.double)
            # edge = edge.astype(np.double)
            return kspace, maps

    @staticmethod
    def getslicefrom_skmtea_xky_imgandseg(fname, slice):

        
        with h5py.File(fname, 'r') as data:
            img_ = data['img'][..., slice]
            seg_ = data['seg'][..., slice]
            segedge = data['segedge'][..., slice]
            imgedge = data['imgedge'][..., slice]

            # img_ = img_.astype(np.double)
            seg_ = seg_.astype(np.double)
            segedge = segedge.astype(np.double)
            imgedge = imgedge.astype(np.double)

            return img_, seg_, segedge, imgedge

    def getslicefrom_skmtea_xky_multicoil(shotname, slice, multicoil_root_dir):
        # multicoil_root_dir = r"./datasets/skmteah5/files_recon_calib-24/all-h5py/"
        # multicoil_root_dir = r"./datasets/skmteah5/files_recon_calib-24/training-h5py/"

        image_path = os.path.join(multicoil_root_dir, shotname + '.h5')

        with h5py.File(image_path, 'r') as data:
            # kspace = data['kspace'][slice][:,:,0,:]  # (512, 512, 160, 8)
            # maps = data['maps'][slice][:,:,:,0]
            multicoil_img = data['kspace'][:,:,slice,:]  # (512, 512, 160, 8)
            maps = data['maps'][:,:,slice,:] 

            # edge = data['edge'][slice]
            # img_ = img_.astype(np.double)
            # seg_ = seg_.astype(np.double)
            # edge = edge.astype(np.double)
            return multicoil_img, maps

    @staticmethod
    def getslicefrom_skmtea_imgandsegandmulticoil(fname, slice):
        multicoil_root_dir = r"./datasets/skmteah5/files_recon_calib-24/all-h5py/"
        image_path = os.path.join(multicoil_root_dir, fname)
        
        with h5py.File(fname, 'r') as data:
            img_ = data['img'][slice]
            seg_ = data['seg'][slice]
            segedge = data['segedge'][slice]
            imgedge = data['imgedge'][slice]

            img_ = img_.astype(np.double)
            seg_ = seg_.astype(np.double)
            segedge = segedge.astype(np.double)
            imgedge = imgedge.astype(np.double)

        with h5py.File(image_path, 'r') as data:
            kspace = data['kspace'][slice]
            maps = data['maps'][slice]
            # edge = data['edge'][slice]
            # img_ = img_.astype(np.double)
            # seg_ = seg_.astype(np.double)
            # edge = edge.astype(np.double)
            # return kspace, maps

            return img_, seg_, segedge, imgedge, 

    @staticmethod
    def normalize_multicoil_to01(k_space):
        # input: k-space multi-coil
        # Shape: (ky, kz, #coils)

        # num_slices = k_space.shape[-1]
        
        # normalized_k_space = np.zeros_like(k_space)
        
        # for slice in range(num_slices):
        #     # Get k-space data for the current coil
        #     k_space_slice = k_space[slice, ...]
            
            # Normalize the k-space data to [0, 1]

        dim = (0,1)

        norm = 'ortho'
        # k_space = ifftshift(k_space, dim=dim)
        imag = np.fft.ifftn(np.fft.ifftshift(k_space, axes=dim), axes=dim)
        # imag = torch.fft.ifftn(ifftshift(k_space, dim=dim), dim=dim)

        max_magnitude = np.max(np.abs(imag))
        k_space = k_space / max_magnitude
    
        # k_space = np.fft.fftshift(np.fft.fftn(imag, axes=dim))

        return k_space

class OASI1_MRB(object):

    @staticmethod
    def get_oasi1mrb_edge_h5py():
        # raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/MRBrainS13DataNii/TrainingData/'
        # https://blog.csdn.net/weixin_40313940/article/details/105799626 标签不对齐。
        # raw_path = r'/home/harddisk8T/wzw/datasets/MICCAI2012/release/MICCAI-2012-Multi-Atlas-Challenge-Data/' # harddist8T
        raw_path = r'/home/wzw/wzw/datasets/OASI1_MRB/'
        # train_img_path = glob.glob(raw_path+'training-h5py/*.h5')
        # test_img_path = glob.glob(raw_path+'testing-h5py/*.h5')
        train_img_path = glob.glob(raw_path+'training-h5py/*.h5')
        # test_img_path = glob.glob(raw_path+'testing-h5py/*1[0-9].nii.h5')
        test_img_path = glob.glob(raw_path+'testing-h5py/*.h5')

        return train_img_path, test_img_path


class Prostate(object):

    @staticmethod
    def get_prostate_edge_h5py():
        # raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/MRBrainS13DataNii/TrainingData/'
        # https://blog.csdn.net/weixin_40313940/article/details/105799626 标签不对齐。
        # raw_path = r'/home/harddisk8T/wzw/datasets/MICCAI2012/release/MICCAI-2012-Multi-Atlas-Challenge-Data/' # harddist8T
        raw_path = r'/home/labuser1/wzw/small_datasets/Prostate/BMC_two_light_clear/h5py'
        # train_img_path = glob.glob(raw_path+'training-h5py/*.h5')
        # test_img_path = glob.glob(raw_path+'testing-h5py/*.h5')
        train_img_path = glob.glob(raw_path+'/training-h5py/*.h5')
        # test_img_path = glob.glob(raw_path+'testing-h5py/*1[0-9].nii.h5')
        test_img_path = glob.glob(raw_path+'/test-h5py/*.h5')

        return train_img_path, test_img_path

class SKMTEA(object):

    @staticmethod
    def get_knee_doubleedge_h5py():
        # raw_path = r'/home/labuser1/wzw/COMBINE_PYTORCH/data/MRBrainS/MRBrainS13DataNii/TrainingData/'
        # https://blog.csdn.net/weixin_40313940/article/details/105799626 标签不对齐。
        # raw_path = r'/home/harddisk8T/wzw/datasets/MICCAI2012/release/MICCAI-2012-Multi-Atlas-Challenge-Data/' # harddist8T
        # raw_path = r'/home/wzw/wzw/datasets/skmteah5'
        raw_path = r'./datasets/skmteah5'

        train_img_path = glob.glob(raw_path+'/training-h5py/*.h5')
        test_img_path = glob.glob(raw_path+'/testing-h5py/*.h5')

        return train_img_path, test_img_path



if __name__ == "__main__":
    # mrB
    path1 = MRB.get_MRBrains18_raw()
    path2 = MRB.get_MRBrains13_raw()
    path3 = MRB.merge13and18()

    # OAI

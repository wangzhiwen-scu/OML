import sys
import numpy as np
import warnings
import random
import h5py
import os

import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from PIL import Image

sys.path.append('.') 
from utils.visdom_visualizer import VisdomLinePlotter
from data.toolbox import get_h5py_shotname
from data.toolbox import OASI1_MRB, IXI_T1_motion, MR_ART, CC_Data2, fastMRI_Dataset, fastMRI_Dataset_lbw

from data.toolbox_v2 import OAI, Miccai

from tqdm import tqdm
import nibabel as nib
from data.bezier_curve import nonlinear_transformation




warnings.filterwarnings("ignore")

class NonLinearTransform:
    def __init__(self, medical_image=False):
        self.medical_image = medical_image

    def __call__(self, img):
        # Assuming nonlinear_transformation is a predefined function 
        # that takes an image and a boolean argument medical_image
        # and returns a list of transformed images.
        return random.choice(nonlinear_transformation(img, self.medical_image))



class DIV2KDataset(Dataset):
    def __init__(self, root_dir, transform=None, bezier=False):
        # Define the transformations you want to apply to the images

        # nonlinear_transform = NonLinearTransform(medical_image=False)

        if transform is None:
            transform = transforms.Compose([
                transforms.RandomCrop(240),
                transforms.RandomHorizontalFlip(), # Flip the image horizontally with a probability of 0.5
                transforms.RandomVerticalFlip(), # Flip the image vertically with a probability of 0.5
                transforms.RandomRotation(15), # Rotate the image by a random angle between -15 to 15 degrees
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # Randomly change the brightness, contrast and saturation of an image
                transforms.ToTensor(),
            ])
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._get_samples()
        # self.samples = self.samples[0]
        self.bezier = bezier
        self.crop_size = 240

    def _get_samples(self):
        samples = []
        for filename in os.listdir(self.root_dir):
            sample_path = os.path.join(self.root_dir, filename)
            samples.append(sample_path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = self.samples[index]
        image = Image.open(image_path).convert("L")  # Convert to grayscale

        # Perform random crop
        # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        # image = transforms.functional.crop(image, i, j, h, w)

        # if self.bezier:
            # image = random.choice(nonlinear_transformation(image, medical_image=False))

        if self.transform is not None:
            image = self.transform(image)

        return image, 1, 2



class H5PYMixedSliceData(Dataset): 
    def __init__(self, dataset_name, root=None, validation=False, test=False,seed=42):
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
        # Supervised branch; Total vol: 80
        # elif dataset_name == 'IXI': 
        train_data_sup, test_data_sup  = IXI_T1_motion.get_ixi_t1_motion_h5py()
        self.getslicefromh5py_Sup = IXI_T1_motion.getslicefromh5py
        paired_files_sup = train_data_sup

        self.examples_Sup = []
        print('Loading dataset :', root)
        random.seed(seed)

        for fname in paired_files_sup:
            h5f = h5py.File(fname, 'r')
            dataimg = h5f['still']

            shotname = get_h5py_shotname(fname)
            num_slices = dataimg.shape[0] 
            self.examples_Sup += [(fname, shotname, slice) for slice in range(num_slices)] 

        #! writ it in one examples.... https://blog.csdn.net/wunianwn/article/details/126965641
        
        # Unsupervised branch, Total vol: 80, same with Supervised branch.
        if dataset_name == 'OASI1_MRB':
            train_data, test_data  = OASI1_MRB.get_oasi1mrb_edge_h5py()
            self.getslicefromh5py_Unsup = OASI1_MRB.getslicefromh5py
        elif dataset_name == 'MR_ART':
            train_data, test_data  = MR_ART.get_mr_art_h5py()
            self.getslicefromh5py_Unsup = MR_ART.getslicefromh5py

        if validation:
            test_data = test_data[0:1]
            
        if root == None:
            paired_files = train_data
        if validation:
            paired_files = test_data
        if test:
            paired_files = test_data



        self.examples_Unsup = []

        for fname in paired_files:
            h5f = h5py.File(fname, 'r')
            dataimg = h5f['standard']

            shotname = get_h5py_shotname(fname)
            num_slices = dataimg.shape[0] 
            self.examples_Unsup += [(fname, shotname, slice) for slice in range(num_slices)] 

        if test:
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
                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(p=0.5)
                    
                ]
            ) 
            self.target_transform = transforms.Compose([
                    
                    transforms.ToTensor(),
                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(p=0.5)
                    
                ]
            )

    def __len__(self):
        return len(self.examples_Sup)

    def __getitem__(self, i):

        fname, shotname, slice = self.examples_Sup[i]
        still, with_motion = self.getslicefromh5py_Sup(fname, slice)

        fname, shotname, slice = self.examples_Unsup[i]
        headmotion1, headmotion2, standard = self.getslicefromh5py_Unsup(fname, slice)

        still = still.astype(np.float32)
        with_motion = with_motion.astype(np.float32)

        # headmotion1 = headmotion1.astype(np.float32)
        headmotion2 = headmotion2.astype(np.float32)
        standard = standard.astype(np.float32)

        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed) 

        still = self.transform(still)
        with_motion = self.transform(with_motion)

        headmotion2 = self.transform(headmotion2)
        # still = self.transform(still)
        standard = self.transform(standard)

        return still, with_motion, headmotion2, standard, shotname


class H5PY_Supervised_Data(Dataset): 
    def __init__(self, dataset_name, change_isotopic=False, root=None, validation=False, test=False, tta_testing=False, bezier=False, seed=42):
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
        # Supervised branch; Total vol: 80
        self.tta_dataset_name = dataset_name
        self.bezier = bezier
        if dataset_name == 'IXI':
            train_data, test_data_org  = IXI_T1_motion.get_ixi_t1_motion_h5py()
            self.getslicefromh5py = IXI_T1_motion.getslicefromh5py

        elif dataset_name[:3] == 'ixi' or dataset_name[:3] == 'sta' or dataset_name[:3] == 'fas' or dataset_name[:3] == 'mrb' or \
            dataset_name[:4] == 'andi':
            train_data, test_data_org = IXI_T1_motion.get_ixi_t1_motion_h5py(subdir=dataset_name)
            if dataset_name == 'ixi_t1_periodic_slight_coronal_xxx': # out-of-fasion
                self.getslicefromh5py = IXI_T1_motion.getslicefromh5py_coronal # [:,slice,:]
            elif dataset_name == 'ixi_t1_periodic_slight_sagittal_xxx':  # out-of-fasion
                self.getslicefromh5py = IXI_T1_motion.getslicefromh5py_sagittal # [:,:,slice]
            else:
                self.getslicefromh5py = IXI_T1_motion.getslicefromh5py # [slice,:,:]

            # if dataset_name != 'ixi_t1_periodic_slight':
            #     train_data = test_data_org
            # elif dataset_name == 'ixi_t1_periodic_slight' and tta_testing:
            #     train_data = test_data_org

            # if dataset_name != 'ixi_t1_periodic_slight':
                # train_data = test_data_org
                
        if tta_testing:
            train_data = test_data_org[5:10]

        if validation:
            test_data = test_data_org[5:10]
            # test_data = test_data_org
            
        if root == None:
            # paired_files = train_data[0:24]
            paired_files = train_data
        if validation:
            paired_files = test_data
        if test:
            paired_files = test_data_org
            # paired_files = test_data

        self.examples = []
        print('Loading dataset :', root)
        random.seed(seed)

        if paired_files: # for paired_files = []
            for fname in paired_files:
                h5f = h5py.File(fname, 'r')
                dataimg = h5f['still']

                # start_slice = np.random.randint(140, 180)
                # end_slice = start_slice + 10

                shotname = get_h5py_shotname(fname)
                if dataset_name == 'ixi_t1_periodic_slight_coronal_xxx':
                    num_slices = dataimg.shape[1]
                elif dataset_name == 'ixi_t1_periodic_slight_sagittal_xxx':
                    num_slices = dataimg.shape[2]
                else:
                    num_slices = dataimg.shape[0] 
                # self.examples += [(fname, shotname, slice) for slice in range(start_slice, end_slice)] 
                self.examples += [(fname, shotname, slice) for slice in range(num_slices)] 

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
                    # transforms.RandomRotation(15),
                    # transforms.RandomHorizontalFlip(p=0.5)
                    
                ]
            ) 
            self.target_transform = transforms.Compose([
                    
                    transforms.ToTensor(),
                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(p=0.5)
                    
                ]
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):

        fname, shotname, slice = self.examples[i]
        still, with_motion = self.getslicefromh5py(fname, slice)

        still = still.astype(np.float32)
        # still = cv2.resize(still, [240, 240])
        # still = cv2.rotate(still, cv2.ROTATE_90_CLOCKWISE)
        still = 1.0 * (still - np.min(still)) / (np.max(still) - np.min(still))
        # still = still/(np.max(still) - np.min(still))

        # noise = np.random.normal(0, 0.02, still.shape)
        # with_noise = still + noise

        # with_noise = 1.0 * (with_noise - np.min(with_noise)) / (np.max(with_noise) - np.min(with_noise))
        # with_motion = with_noise

        with_motion = with_motion.astype(np.float32)
        # with_motion = cv2.rotate(with_motion, cv2.ROTATE_90_CLOCKWISE)
        # with_motion = cv2.resize(with_motion, [240, 240])
        with_motion = 1.0 * (with_motion - np.min(with_motion)) / (np.max(with_motion) - np.min(with_motion))
        # with_motion = with_motion/(np.max(with_motion) - np.min(with_motion))

        if self.tta_dataset_name[:6] == 'ixi_t2' or self.tta_dataset_name[:6] == "ixi_pd" \
            or self.tta_dataset_name == 'stanford_knee_axial_pd_periodic_slight' or self.tta_dataset_name[:15] == 'fastmribrain_t1' \
            or self.tta_dataset_name == 'mrb13_t1_sudden_sslight':

            still = cv2.resize(still, [240, 240])
            with_motion = cv2.resize(with_motion, [240, 240])
        if self.tta_dataset_name[:6] == 'ixi_t2' or self.tta_dataset_name[:6] == "ixi_pd":
            still = cv2.rotate(still, cv2.ROTATE_180)
            with_motion = cv2.rotate(with_motion, cv2.ROTATE_180)
        
        # if self.tta_dataset_name[:6] == 'ixi_t2' and self.bezier:
        if self.bezier:
            still = random.choice(nonlinear_transformation(still))
            with_motion = random.choice(nonlinear_transformation(with_motion))

        if self.tta_dataset_name == 'stanford_knee_axial_pd_periodic_moderate' or  \
             self.tta_dataset_name == 'ixi_t1_periodic_slight_rl':
            still = cv2.rotate(still, cv2.ROTATE_90_COUNTERCLOCKWISE)
            with_motion = cv2.rotate(with_motion, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.tta_dataset_name[:15] == 'fastmribrain_t1':
            still = cv2.rotate(still, cv2.ROTATE_180)
            with_motion = cv2.rotate(with_motion, cv2.ROTATE_180)

        # if self.tta_dataset_name[:4] == 'andi':
        #     still = cv2.rotate(still, cv2.ROTATE_180)
        #     with_motion = cv2.rotate(with_motion, cv2.ROTATE_180)

            # still = cv2.resize(still, [240, 240])
            # with_motion = cv2.resize(with_motion, [240, 240])

        if self.tta_dataset_name == 'ixi_t1_periodic_slight_coronal' or self.tta_dataset_name == 'ixi_t1_periodic_slight_sagittal' or \
            self.tta_dataset_name == 'ixi_t1_periodic_moderate_coronal' or self.tta_dataset_name == 'ixi_t1_periodic_moderate_sagittal':
            still = cv2.rotate(still, cv2.ROTATE_180)
            with_motion = cv2.rotate(with_motion, cv2.ROTATE_180)
            
            still = cv2.flip(still, 1) # 1 is vertically flip
            with_motion = cv2.flip(with_motion, 1)

            # still = cv2.rotate(still, cv2.ROTATE_90_CLOCKWISE)
            # with_motion = cv2.rotate(with_motion, cv2.ROTATE_90_CLOCKWISE)

        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed) 
        still = self.transform(still)
        random.seed(seed) 
        torch.manual_seed(seed) 
        with_motion = self.transform(with_motion)

        return still, with_motion, shotname

class H5PY_Unsupervised_Data(Dataset): 
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
        self.slight_motion = slight_motion
        # Supervised branch; Total vol: 80
        self.change_isotopic = change_isotopic
        if dataset_name == 'OASI1_MRB':
            train_data, test_data  = OASI1_MRB.get_oasi1mrb_edge_h5py()
            self.getslicefromh5py_Unsup = OASI1_MRB.getslicefromh5py
        elif dataset_name == 'MR_ART':
            train_data, test_data_orig  = MR_ART.get_mr_art_h5py()
            self.getslicefromh5py_Unsup = MR_ART.getslicefromh5py
            
        elif dataset_name[:3] == 'ixi':
            train_data, test_data_orig = IXI_T1_motion.get_ixi_t1_motion_h5py(subdir=dataset_name)
            self.getslicefromh5py_Unsup = IXI_T1_motion.getslicefromh5py

        if validation:
            test_data = test_data_orig[0:1]
            
        if root == None:
            paired_files = train_data
            if one_shot:
                paired_files = train_data[0:1]

        if validation:
            paired_files = test_data
        if test:
            paired_files = test_data_orig



        self.examples = []

        for fname in paired_files:
            h5f = h5py.File(fname, 'r')
            dataimg = h5f['standard']

            shotname = get_h5py_shotname(fname)
            num_slices = dataimg.shape[0] 
            self.examples += [(fname, shotname, slice) for slice in range(num_slices)] 

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
                    transforms.RandomRotation(5),
                    transforms.RandomHorizontalFlip(p=0.1)
                    
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
        headmotion1, headmotion2, standard = self.getslicefromh5py_Unsup(fname, slice)

        if self.slight_motion:
            headmotion2 = headmotion1
        # headmotion1 = headmotion1.astype(np.float32)
        headmotion2 = headmotion2.astype(np.float32)
        headmotion2 = cv2.rotate(headmotion2, cv2.ROTATE_180)
        if self.change_isotopic:
            # headmotion2 = cv2.resize(headmotion2, [384, 240])
            headmotion2 = self._get_240img(headmotion2)
        headmotion2 = 1.0 * (headmotion2 - np.min(headmotion2)) / (np.max(headmotion2) - np.min(headmotion2))
        if not self.change_isotopic:
            headmotion2 = self._get_240img(headmotion2)

        standard = standard.astype(np.float32)
        standard = cv2.rotate(standard, cv2.ROTATE_180)
        if self.change_isotopic:
            # standard = cv2.resize(standard, [384, 240])
            standard = self._get_240img(standard)

        standard = 1.0 * (standard - np.min(standard)) / (np.max(standard) - np.min(standard))
        if not self.change_isotopic:
            standard = self._get_240img(standard)
        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed) 
        headmotion2 = self.transform(headmotion2)
        # still = self.transform(still)
        random.seed(seed) 
        torch.manual_seed(seed) 
        standard = self.transform(standard)

        return standard, headmotion2, shotname

    def _get_240img(self, input_img):
        offfset256to240 = input_img.shape[0]  - 240
        input_img = input_img[offfset256to240-1:-1, :]
        input_img = input_img[:,:, np.newaxis] # (256,192) -> (256,192,1)
        x_offset = input_img.shape[1] - 240
        y_offset = input_img.shape[1] - 240
        offset = (int(abs(x_offset/2)), int(abs(y_offset/2)))
        npad = ((0, 0), offset, (0, 0)) # (240,180,1) -> (240,240,1)
        output = np.pad(input_img, pad_width=npad, mode='constant', constant_values=0)  # (z, x, y) 
        # input_img = np.square(input_img) # (240,180) -> (240,180,1)
        # output = cv2.resize(output, [240, 240])
        return output

def add_gaussian_noise(image_tensor, sigma=0.1):
    noise = torch.randn(image_tensor.size()) * sigma
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0, 1)
    return noisy_image


class TTA_MRIDataset(Dataset): 
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

        # Supervised branch; Total vol: 80
        self.dataset_name = dataset_name
        self.change_isotopic = change_isotopic
        if dataset_name == 'cc_data2_brain_t1':
            train_data, test_data  = CC_Data2.get_path_lists()
            self.getslicefromNpy = CC_Data2.getslicefromNpy
        elif dataset_name == 'fastmri_brain_t1':
            train_data, test_data  = fastMRI_Dataset.get_t1_path_lists() # train500, test80
            train_data = train_data[0:70]
            self.getslicefromNpy = fastMRI_Dataset.getslicefromNpy
        elif dataset_name == 'fastmri_brain_t2':
            train_data, test_data  = fastMRI_Dataset.get_t2_path_lists()
            self.getslicefromNpy = fastMRI_Dataset.getslicefromNpy
            train_data = train_data[0:70]
            # test_data = test_data[0:1]
        elif dataset_name == 'fastmri_knee_pd':
            train_data, test_data  = fastMRI_Dataset.get_pd_knee_path_lists()
            self.getslicefromNpy = fastMRI_Dataset.getslicefromNpyforKnee
            # train_data = train_data[0:20]
        elif dataset_name == 'fastmri_knee_pdfs':
            train_data, test_data  = fastMRI_Dataset.get_pdfs_knee_path_lists()
            self.getslicefromNpy = fastMRI_Dataset.getslicefromNpyforKnee
        elif dataset_name == 'ixi_t1_periodic_slight_sagittal':
            train_data, test_data = IXI_T1_motion.get_ixi_t1_motion_h5py_tta(subdir=dataset_name)
            self.getslicefromNpy = IXI_T1_motion.getslicefromh5pyfortta # [:,:,slice]


        elif dataset_name == 'fastmri_brain_t1_lbw':
            train_data, test_data  = fastMRI_Dataset_lbw.get_path_lists()
            self.getslicefromNpy = fastMRI_Dataset_lbw.getslicefromNpy
        elif dataset_name == 'fastmri_brain_t2_lbw':
            train_data, test_data  = fastMRI_Dataset_lbw.get_t2_path_lists()
            self.getslicefromNpy = fastMRI_Dataset_lbw.getslicefromNpy

        # if validation:
            # test_data = test_data_orig[0:1]
            
        if root == None:
            paired_files = train_data
            if one_shot:
                paired_files = train_data[0:1]

        if validation:
            paired_files = test_data
        if test:
            # paired_files = test_data_orig
            pass



        self.examples = []

        for fname in paired_files:


            start_slice = 0
            if dataset_name == 'cc_data2_brain_t1':
                dataimg= np.load(fname)
                num_slices = dataimg.shape[-1]
                # end_slice = num_slices  # 90 slices.
                start_slice = 40
                end_slice = 70

            elif dataset_name == "ixi_t1_periodic_slight_sagittal":
                h5f = h5py.File(fname, 'r')
                dataimg = h5f['still']
                num_slices = dataimg.shape[0]
                end_slice = num_slices

            else:
                h5f = h5py.File(fname, 'r')
                dataimg = h5f['reconstruction_esc']
                num_slices = dataimg.shape[0]



                if dataset_name == 'fastmri_knee_pd' or dataset_name == 'fastmri_knee_pdfs':
                    start_slice = 15
                    end_slice = num_slices-10

                else:
                    start_slice = 5
                    end_slice = num_slices-5

                # if dataset_name == 'fastmri_brain_t2':
                #     start_slice = 5
                #     end_slice = 6

            shotname = get_h5py_shotname(fname)
            
            self.examples += [(fname, shotname, slice) for slice in range(start_slice, end_slice)] 

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
        # if self.change_isotopic:
            # headmotion2 = self._get_240img(headmotion2)
        # headmotion2 = 1.0 * (headmotion2 - np.min(headmotion2)) / (np.max(headmotion2) - np.min(headmotion2))

        img = cv2.resize(img, [256, 256])
        # img = cv2.resize(img, [240, 240])

        if self.dataset_name == "fastmri_brain_t1" or self.dataset_name == "fastmri_brain_t2":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.dataset_name == "ixi_t1_periodic_slight_sagittal":
            img = cv2.rotate(img, cv2.ROTATE_180)            
            img = cv2.flip(img, 1) # 1 is vertically flip

        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed)
        img = self.transform(img)


        img_max = torch.max(torch.abs(img))
        img = img / img_max

        if self.dataset_name == "cc_data2_brain_t1":
            img = torch.abs(img)

        return img, 1, shotname

    def _get_240img(self, input_img):
        offfset256to240 = input_img.shape[0]  - 240
        input_img = input_img[offfset256to240-1:-1, :]
        input_img = input_img[:,:, np.newaxis] # (256,192) -> (256,192,1)
        x_offset = input_img.shape[1] - 240
        y_offset = input_img.shape[1] - 240
        offset = (int(abs(x_offset/2)), int(abs(y_offset/2)))
        npad = ((0, 0), offset, (0, 0)) # (240,180,1) -> (240,240,1)
        output = np.pad(input_img, pad_width=npad, mode='constant', constant_values=0)  # (z, x, y) 
        # input_img = np.square(input_img) # (240,180) -> (240,180,1)
        # output = cv2.resize(output, [240, 240])
        return output
    

class TTA_MRIDataset_Noise(Dataset): 
    def __init__(self, dataset_name, sigma, change_isotopic=True, root=None, validation=False, test=False, slight_motion=False, one_shot=False, seed=42):
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

        # Supervised branch; Total vol: 80
        self.dataset_name = dataset_name
        self.change_isotopic = change_isotopic
        self.noise_sigma = sigma
        if dataset_name == 'cc_data2_brain_t1':
            train_data, test_data  = CC_Data2.get_path_lists()
            self.getslicefromNpy = CC_Data2.getslicefromNpy
        elif dataset_name == 'fastmri_brain_t1':
            train_data, test_data  = fastMRI_Dataset.get_t1_path_lists() # train500, test80
            train_data = train_data[0:70]
            self.getslicefromNpy = fastMRI_Dataset.getslicefromNpy
        elif dataset_name == 'fastmri_brain_t2':
            train_data, test_data  = fastMRI_Dataset.get_t2_path_lists()
            self.getslicefromNpy = fastMRI_Dataset.getslicefromNpy
            train_data = train_data[0:70]
            # test_data = test_data[0:1]
        elif dataset_name == 'fastmri_knee_pd':
            train_data, test_data  = fastMRI_Dataset.get_pd_knee_path_lists()
            self.getslicefromNpy = fastMRI_Dataset.getslicefromNpyforKnee
            # train_data = train_data[0:20]
        elif dataset_name == 'fastmri_knee_pdfs':
            train_data, test_data  = fastMRI_Dataset.get_pdfs_knee_path_lists()
            self.getslicefromNpy = fastMRI_Dataset.getslicefromNpyforKnee
        elif dataset_name == 'ixi_t1_periodic_slight_sagittal':
            train_data, test_data = IXI_T1_motion.get_ixi_t1_motion_h5py_tta(subdir=dataset_name)
            self.getslicefromNpy = IXI_T1_motion.getslicefromh5pyfortta # [:,:,slice]


        elif dataset_name == 'fastmri_brain_t1_lbw':
            train_data, test_data  = fastMRI_Dataset_lbw.get_path_lists()
            self.getslicefromNpy = fastMRI_Dataset_lbw.getslicefromNpy
        elif dataset_name == 'fastmri_brain_t2_lbw':
            train_data, test_data  = fastMRI_Dataset_lbw.get_t2_path_lists()
            self.getslicefromNpy = fastMRI_Dataset_lbw.getslicefromNpy

        # if validation:
            # test_data = test_data_orig[0:1]
            
        if root == None:
            paired_files = train_data
            if one_shot:
                paired_files = train_data[0:1]

        if validation:
            paired_files = test_data
        if test:
            # paired_files = test_data_orig
            pass



        self.examples = []

        for fname in paired_files:


            start_slice = 0
            if dataset_name == 'cc_data2_brain_t1':
                dataimg= np.load(fname)
                num_slices = dataimg.shape[-1]
                # end_slice = num_slices  # 90 slices.
                start_slice = 40
                end_slice = 70

            elif dataset_name == "ixi_t1_periodic_slight_sagittal":
                h5f = h5py.File(fname, 'r')
                dataimg = h5f['still']
                num_slices = dataimg.shape[0]
                end_slice = num_slices

            else:
                h5f = h5py.File(fname, 'r')
                dataimg = h5f['reconstruction_esc']
                num_slices = dataimg.shape[0]



                if dataset_name == 'fastmri_knee_pd' or dataset_name == 'fastmri_knee_pdfs':
                    start_slice = 15
                    end_slice = num_slices-10

                else:
                    start_slice = 5
                    end_slice = num_slices-5

                # if dataset_name == 'fastmri_brain_t2':
                #     start_slice = 5
                #     end_slice = 6

            shotname = get_h5py_shotname(fname)
            
            self.examples += [(fname, shotname, slice) for slice in range(start_slice, end_slice)] 

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
        # if self.change_isotopic:
            # headmotion2 = self._get_240img(headmotion2)
        # headmotion2 = 1.0 * (headmotion2 - np.min(headmotion2)) / (np.max(headmotion2) - np.min(headmotion2))

        img = cv2.resize(img, [256, 256])
        # img = cv2.resize(img, [240, 240])

        if self.dataset_name == "fastmri_brain_t1" or self.dataset_name == "fastmri_brain_t2":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.dataset_name == "ixi_t1_periodic_slight_sagittal":
            img = cv2.rotate(img, cv2.ROTATE_180)            
            img = cv2.flip(img, 1) # 1 is vertically flip

        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed)
        img = self.transform(img)


        img_max = torch.max(torch.abs(img))
        img = img / img_max

        # Add Gaussian noise
        noisy_image_tensor = add_gaussian_noise(img, sigma=self.noise_sigma)

        if self.dataset_name == "cc_data2_brain_t1":
            img = torch.abs(img)

        return img, noisy_image_tensor, shotname

    def _get_240img(self, input_img):
        offfset256to240 = input_img.shape[0]  - 240
        input_img = input_img[offfset256to240-1:-1, :]
        input_img = input_img[:,:, np.newaxis] # (256,192) -> (256,192,1)
        x_offset = input_img.shape[1] - 240
        y_offset = input_img.shape[1] - 240
        offset = (int(abs(x_offset/2)), int(abs(y_offset/2)))
        npad = ((0, 0), offset, (0, 0)) # (240,180,1) -> (240,240,1)
        output = np.pad(input_img, pad_width=npad, mode='constant', constant_values=0)  # (z, x, y) 
        # input_img = np.square(input_img) # (240,180) -> (240,180,1)
        # output = cv2.resize(output, [240, 240])
        return output


class H5PYMixedBackGroundSliceData_withmultcoilandsense(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    # https://github.com/guopengf/FL-MRCM/blob/main/data/mri_data.py
    """
    def __init__(self, dataset_name, root=None, validation=False, test=False,seed=42):
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
        if dataset_name == 'MICCAI':
            train_data, test_data  = Miccai.get_Miccai_h5py()
            # train_data= train_data[0:20]
            test_data = test_data[0:10]
            pass
        elif dataset_name == 'OAI':
            train_data, test_data  = OAI.get_oai_edge_h5py()
            train_data= train_data[0:20]
            test_data = test_data[0:10]

            # train_data = train_data[0:50]
        # for not wasting time of training, we take [0:1] of test_dataset.
        elif dataset_name == 'OASI1_MRB':
            train_data, test_data  = OASI1_MRB.get_oasi1mrb_edge_h5py()
        
        
        if validation:
            # train_data= train_data[0:10]
            test_data = test_data[0:1]
            
        if root == None:
            paired_files = train_data
        if validation:
            paired_files = test_data
        if test:
            paired_files = test_data

        # dataset_name = root.parts[-3]
        self.examples = []
        # files = list(pathlib.Path(root).iterdir())
        print('Loading dataset :', root)
        random.seed(seed)

        for fname in paired_files:
            h5f = h5py.File(fname, 'r')
            dataimg = h5f['img']

            shotname = get_h5py_shotname(fname)
            num_slices = dataimg.shape[0] # img_z, img_x, img_y = x_train.shape
            self.examples += [(fname, shotname, slice) for slice in range(num_slices)] # ((img_file, seg_file), shotname, slice)

        if test:
            self.transform = transforms.Compose([
                    # transforms.Resize((256, 256)),
                    transforms.ToTensor()
                    # transforms.Normalize((0.5), (0.5))
                ]
            )    # 不加totensor是[4, 256, 256], 加了是[4, 1, 256, 256]
            self.target_transform = transforms.Compose([
                    # transforms.Resize((256, 256)),
                    transforms.ToTensor()
                    # transforms.Normalize((0.5), (0.5))
                ]
            )    # 不加totensor是[4, 256, 256], 加了是[4, 1, 256, 256]            
        else:
            self.transform = transforms.Compose([
                    # transforms.Resize((256, 256)),
                    transforms.ToTensor(),

                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(p=0.5)

                    # transforms.Normalize((0.5), (0.5))
                ]
            )    # 不加totensor是[4, 256, 256], 加了是[4, 1, 256, 256]
            self.target_transform = transforms.Compose([
                    # transforms.Resize((256, 256)),
                    transforms.ToTensor(),

                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(p=0.5)
                    
                    # transforms.Normalize((0.5), (0.5))
                ]
            )    # 不加totensor是[4, 256, 256], 加了是[4, 1, 256, 256]        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, shotname, slice = self.examples[i]
        img_, seg_, edge = Miccai.getslicefromh5py(fname, slice)  # (240, 240) astype(np.uint8)

        y_train = seg_.transpose((1,2,0))  #! channel must be last. (H,W,C)
        # y_edge = edge.transpose((1,2,0))  #! channel must be last. (H,W,C)
        y_edge = edge

        # epsilon=1e-8
        x_train = img_.astype(np.float32) # np.float32
        x_train = x_train / (np.max(x_train) -np.min(x_train))

        # x_train = 255 * (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))

        y_train = y_train.astype(np.float32)
        y_background = np.sum(y_train, axis=-1, keepdims=True) # ! here.
        y_background = y_background.astype(np.bool)
        y_background = y_background.astype(np.float32)
        y_train = np.concatenate((y_train, y_background), axis=-1) #

        y_edge = y_edge.astype(np.float32) # np.double

        y_edge_ = y_edge[:,:, np.newaxis]
       
        y_train = np.concatenate((y_train, y_edge_), axis=-1) #

        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        # if self.transform is not None:
        x_train_org = self.transform(x_train)
        # x_train = simulate_multicoil_froIMG(x_train_org)
        x_train_org = x_train_org
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        y_train = self.transform(y_train)
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        y_edge = self.transform(y_edge)
        return x_train, y_train, y_edge, shotname

if __name__ == '__main__':
    import time
    dataset_name_lists = ['cc_data2_brain_t1', 'fastmri_brain_t1', 'fastmri_brain_t2', 
                          'fastmri_knee_pd', 'fastmri_knee_pdfs', 'ixi_t1_periodic_slight_sagittal']
    batch = 1
    # dataset = MiccaiSliceData() # 
    # dataset = MemoryMiccaiSliceData() # 很卡。
    # dataset_sup = TTA_MRIDataset(dataset_name=dataset_name_lists[2]) # ixi_t1_periodic_slight_sagittal
    dataset_sup = TTA_MRIDataset_Noise(dataset_name="fastmri_brain_t1", sigma=0.02) # ixi_t1_periodic_slight_sagittal

    dataloader_sup = DataLoader(dataset_sup, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)  # 增加GPU利用率稳定性。

    # dataset_unsup = H5PY_Unsupervised_Data(dataset_name='MR_ART')
    # dataloader_unsup = DataLoader(dataset_unsup, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)  # 增加GPU利用率稳定性。

    step = 0
    plotter = VisdomLinePlotter(env_name='test')

    # loop = tqdm(enumerate(zip(dataloader_sup, dataloader_unsup)), total=len(dataloader_sup))

    # for index, x in loop:
    # for x in dataloader_sup:
    loop = tqdm(enumerate(dataloader_sup), total=len(dataloader_sup))
    for index, (still, with_motion, shotname) in loop:
        step += 1

        # img, _temp, shotname = x

        # for cc.

        # plotter.k_space_clip(shotname[0], step, img)
        # plotter.image(shotname[0], step, torch.abs(img))

        # standard, headmotion2, shotname = x[1]

        # if [still.shape[2], still.shape[3]] != [240,240]:
        #     print(still.shape)
        # if [with_motion.shape[2], with_motion.shape[3]] != [240,240]:
        #     print(with_motion.shape)

        plotter.image(shotname[0][0:5]+'clean', step, still)
        plotter.image(shotname[0][0:5]+'noise', step, with_motion)
        # plotter.image(shotname[0][0:5]+'img1_with_motion', step, with_motion)
        # plotter.image(shotname[0][0:5]+'img2_headmotion2', step, headmotion2)
        # plotter.image(shotname[0][0:5]+'img2_standard', step, standard)

        time.sleep(0.1)
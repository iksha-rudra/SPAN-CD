# Imports

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as tr

# Other
import os
import numpy as np
import random
from skimage import io
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

from tqdm import tqdm as tqdm
from pandas import read_csv
from math import floor, ceil, sqrt, exp
from IPython import display
import time
from itertools import chain
import time
import warnings
from pprint import pprint

print('IMPORTS OK')
FP_MODIFIER = 10
TYPE = 3 # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands

class Sentinel2_OSCD_SSL(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, 
                 path, 
                 transform,
                 fname = 'all.txt', 
                 patch_side = 96, 
                 stride = None, 
                 normalize=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # basics
        self.transform = transform
        self.normalize = normalize
        self.path = path
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride
        
        # print(path + fname)
        self.names = read_csv(path + fname).columns
        self.n_imgs = self.names.shape[0]
        
        n_pix = 0
        true_pix = 0
        
        
        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            # load and store each image
            I1, I2, cm = self.read_sentinel_img_trio(self.path + im_name)
            self.imgs_1[im_name] = I1
            self.imgs_2[im_name] = I2
            self.change_maps[im_name] = cm
            
            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()
            
            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i
            
            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (im_name, 
                                    [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                    [self.stride*(i + 1), self.stride*(j + 1)])
                    self.patch_coords.append(current_patch_coords)
                    
        self.weights = [ FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]
        
    def get_weights(self):
        return self.weights

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        
        I = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        # I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        # label = self.change_maps[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        
        # sample = {'I1': I1, 'I2': I2, 'label': label, 'fname': im_name}
        
        I1 = I
        I2 = I

        if self.transform:
            I1, I2 = sample = self.transform(I)

        return I1.float(), I2.float()
    
    # Functions
    def get_padded_image(self, image, 
                        patch_size=256,
                        stride=128):
    
        s = image.shape
        
        x = s[0]
        y = s[1]

        if stride > 1:
            pad_x = stride - x % stride
            pad_y = stride - y % stride

        pad_x_1 = 0
        pad_x_2 = 0
        pad_y_1 = 0 
        pad_y_2 = 0

        pad_x_1 = int(0.5 + pad_x/2)
        pad_x_2 = int(pad_x/2)

        pad_y_1 = int(0.5 + pad_y/2)
        pad_y_2 = int(pad_y/2)

        image = np.pad(image, ((pad_x_1,pad_x_2), (pad_y_1,pad_y_2), (0, 0)), 'symmetric')
        
        return image

    def image_normalize_quantile(self,image, first=1, last=99):
        (M,N,C) = image.shape
        for i in range(C):
            chan = image[:,:,i]
            per_1 = np.percentile(chan,first)
            per_99 = np.percentile(chan,99)
            min = per_1
            max = per_99
            chan = (chan - min )/( max - min )
            chan = chan.clip(0,1)
            image[:,:,i] = chan
        return image
        
    def adjust_shape(self, I, s):
        """Adjust shape of grayscale image I to s."""
        
        # crop if necesary
        I = I[:s[0],:s[1]]
        si = I.shape
        
        # pad if necessary 
        p0 = max(0,s[0] - si[0])
        p1 = max(0,s[1] - si[1])
        
        return np.pad(I,((0,p0),(0,p1)),'edge')
    

    def read_sentinel_img(self, path):
        """Read cropped Sentinel-2 image: RGB bands."""
        im_name = os.listdir(path)[0][:-7]
        r = io.imread(path + im_name + "B04.tif")
        g = io.imread(path + im_name + "B03.tif")
        b = io.imread(path + im_name + "B02.tif")
        
        I = np.stack((r,g,b),axis=2).astype('float')
        
        if self.normalize:
            I = self.image_normalize_quantile(I)
            # I = (I-I.min())/(I.max()-I.min())
            # I = (I - I.mean()) / I.std()

        return I

    def read_sentinel_img_4(self, path):
        """Read cropped Sentinel-2 image: RGB and NIR bands."""
        im_name = os.listdir(path)[0][:-7]
        r = io.imread(path + im_name + "B04.tif")
        g = io.imread(path + im_name + "B03.tif")
        b = io.imread(path + im_name + "B02.tif")
        nir = io.imread(path + im_name + "B08.tif")
        
        I = np.stack((r,g,b,nir),axis=2).astype('float')
        
        if self.normalize:
            I = self.image_normalize_quantile(I)
            # I = (I-I.min())/(I.max()-I.min())
            # I = (I - I.mean()) / I.std()

        return I

    def read_sentinel_img_leq20(self, path):
        """Read cropped Sentinel-2 image: bands with resolution less than or equals to 20m."""
        im_name = os.listdir(path)[0][:-7]
        
        r = io.imread(path + im_name + "B04.tif")
        s = r.shape
        g = io.imread(path + im_name + "B03.tif")
        b = io.imread(path + im_name + "B02.tif")
        nir = io.imread(path + im_name + "B08.tif")
        
        ir1 = self.adjust_shape(zoom(io.imread(path + im_name + "B05.tif"),2),s)
        ir2 = self.adjust_shape(zoom(io.imread(path + im_name + "B06.tif"),2),s)
        ir3 = self.adjust_shape(zoom(io.imread(path + im_name + "B07.tif"),2),s)
        nir2 = self.adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"),2),s)
        swir2 = self.adjust_shape(zoom(io.imread(path + im_name + "B11.tif"),2),s)
        swir3 = self.adjust_shape(zoom(io.imread(path + im_name + "B12.tif"),2),s)
        
        I = np.stack((r,g,b,nir,ir1,ir2,ir3,nir2,swir2,swir3),axis=2).astype('float')
        
        if self.normalize:
            I = self.image_normalize_quantile(I)
            # I = (I-I.min())/(I.max()-I.min())
            # I = (I - I.mean()) / I.std()

        return I

    def read_sentinel_img_leq60(self, path):
        """Read cropped Sentinel-2 image: all bands."""
        im_name = os.listdir(path)[0][:-7]
        
        r = io.imread(path + im_name + "B04.tif")
        s = r.shape
        g = io.imread(path + im_name + "B03.tif")
        b = io.imread(path + im_name + "B02.tif")
        nir = io.imread(path + im_name + "B08.tif")
        
        ir1 = self.adjust_shape(zoom(io.imread(path + im_name + "B05.tif"),2),s)
        ir2 = self.adjust_shape(zoom(io.imread(path + im_name + "B06.tif"),2),s)
        ir3 = self.adjust_shape(zoom(io.imread(path + im_name + "B07.tif"),2),s)
        nir2 = self.adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"),2),s)
        swir2 = self.adjust_shape(zoom(io.imread(path + im_name + "B11.tif"),2),s)
        swir3 = self.adjust_shape(zoom(io.imread(path + im_name + "B12.tif"),2),s)
        
        uv = self.adjust_shape(zoom(io.imread(path + im_name + "B01.tif"),6),s)
        wv = self.adjust_shape(zoom(io.imread(path + im_name + "B09.tif"),6),s)
        swirc = self.adjust_shape(zoom(io.imread(path + im_name + "B10.tif"),6),s)
        
        I = np.stack((r,g,b,nir,ir1,ir2,ir3,nir2,swir2,swir3,uv,wv,swirc),axis=2).astype('float')
        
        if self.normalize:
            I = self.image_normalize_quantile(I)
            # I = (I-I.min())/(I.max()-I.min())
            # I = (I - I.mean()) / I.std()

        return I

    def read_sentinel_img_trio(self, path):
        """Read cropped Sentinel-2 image pair and change map."""
    #     read images
        if TYPE == 0:
            I1 = self.read_sentinel_img(path + '/imgs_1/')
            I2 = self.read_sentinel_img(path + '/imgs_2/')
        elif TYPE == 1:
            I1 = self.read_sentinel_img_4(path + '/imgs_1/')
            I2 = self.read_sentinel_img_4(path + '/imgs_2/')
        elif TYPE == 2:
            I1 = self.read_sentinel_img_leq20(path + '/imgs_1/')
            I2 = self.read_sentinel_img_leq20(path + '/imgs_2/')
        elif TYPE == 3:
            I1 = self.read_sentinel_img_leq60(path + '/imgs_1/')
            I2 = self.read_sentinel_img_leq60(path + '/imgs_2/')
            
        # crop if necessary
        s1 = I1.shape
        s2 = I2.shape
        I2 = np.pad(I2,((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0,0)),'edge')
        
        # cm = io.imread(path + '/cm/cm.png', as_gray=True) != 0
        name = path.split('/')[-1]
        cm = io.imread(path + '/cm/'+name+'-cm.tif')
        cm = cm - 1
        cm = np.expand_dims(cm,axis=2)
        
        I1 = self.get_padded_image(I1,self.patch_side,self.stride)
        I2 = self.get_padded_image(I2,self.patch_side,self.stride)
        cm = self.get_padded_image(cm,self.patch_side,self.stride)
        
        I1 = self.reshape_for_torch(I1)    
        I2 = self.reshape_for_torch(I2)
        cm = self.reshape_for_torch(cm)
        
        return I1, I2, cm



    def reshape_for_torch(self, I):
        """Transpose image for PyTorch coordinates."""

        out = I.transpose((2, 0, 1))
        return torch.from_numpy(out)
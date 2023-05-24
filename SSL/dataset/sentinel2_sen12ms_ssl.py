import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import ceil
import os
import numpy as np
import random
from skimage import io

from torch.utils.data import Dataset
from tqdm import tqdm as tqdm

class Sentinel2_SEN12MS_SSL(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, 
                 path = ['/home/rakesh/DataSet/SEN12MS/ROIs1868_summer/'], 
                 fname = '',
                 bands = ['B01','B02','B03','B04','B05',
                                        'B06','B07','B08','B8A','B09',
                                        'B10','B11','B12'],
                 patch_side = 256, 
                 stride = 128,
                 normalize = True, 
                 transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # basics
        self.bands = bands
        self.transform = transform
        self.normalize = normalize
        self.path = path
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride
        
        list = []
        for dir in path:
            for dirname in os.listdir(dir):
                for filename in os.listdir(os.path.join(dir, dirname)):
                    list.append(os.path.join(dir, dirname, filename))

        self.names = list
        self.imgs = []

        for name in tqdm(self.names):
            # I = self.get_sen12ms_image(name)
            # self.imgs.append(I)
            self.imgs.append(name)  
        
    def get_sen12ms_image(self, img_path):
        image = io.imread(img_path)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = torch.einsum('ijk->kij',image)
        return image

    def __len__(self):
        return len(self.imgs)

    def clip_1_99_percentile(self,image, first=0.01, last=0.99):
        
        (C,M,N) = image.shape
        image_np = image.numpy()
        
        for i in range(C):
            a, b = np.percentile(image_np[i, :, :], (1, 99))
            image_np[i, :, :] = np.clip(image_np[i, :, :], a, b)
            
            # image_np[i, :, :] = (image_np[i, :, :] - \
            #     image_np[i, :, :].min()) / \
            #         (image_np[i, :, :].max() - \
            #             image_np[i, :, :].min())
        
        image = torch.from_numpy(image_np)
        return image

    def __getitem__(self, idx):
        I = self.get_sen12ms_image(self.imgs[idx])
        # I = self.imgs[idx]

        if self.normalize:
            pass
            I = self.clip_1_99_percentile(I)
            # I = (I - I.mean()) / I.std()
            # I = (I - I.min()) / (I.max() - I.min())
        
        if self.transform:
            I1, I2 = self.transform(I)
        
        if self.normalize:
            # pass
            I1 = (I1 - I1.mean()) / I1.std()
            I2 = (I2 - I2.mean()) / I2.std()
        
        return (I1, I2)
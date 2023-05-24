import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import ceil
import os
import numpy as np
import random

from torch.utils.data import Dataset

class Sentinel2_S2MTCP_SSL(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, 
                 path, 
                 fname = 'S2MTCP_metadata.csv',
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
        
        file_path = os.path.join(path, fname)

        with open(file_path, 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            list = [row[3] for row in csvreader]

        self.names = list
        self.n_imgs = len(self.names)//2
        
        self.img_pairs = []

        for i in range(0,len(self.names),2):
            im_name1 = self.names[i].strip()
            im_name2 = self.names[i+1].strip()
            
            I1 = self.get_img(im_name1)
            I2 = self.get_img(im_name2)

            img_pairs = (I1, I2)
            self.img_pairs.append(img_pairs)

            # load and store each image
            print(f'reading....{im_name1}')
            print(f'reading....{im_name2}')
            break
        
    def get_s2mtcp_image(self, img_path):
        image = torch.from_numpy(np.load(img_path)).float()
        image = torch.einsum('ijk->kij',image)
        image = image[0:13,:,:]
        return image

    def get_img(self, im_name):
        return self.get_s2mtcp_image(os.path.join(self.path, 'data_S21C', im_name))

    def __len__(self):
        return self.n_imgs

    def clip_1_99_percentile(self,image, first=0.01, last=0.99):
        
        (C,M,N) = image.shape
        image_np = image.numpy()
        
        for i in range(C):
            a, b = np.percentile(image_np[i, :, :], (1, 99))
            image_np[i, :, :] = np.clip(image_np[i, :, :], a, b)
            
            image_np[i, :, :] = (image_np[i, :, :] - \
                image_np[i, :, :].min()) / \
                    (image_np[i, :, :].max() - \
                        image_np[i, :, :].min())
        
        image = torch.from_numpy(image_np)
        return image

    def random_crop(self, pre_img, post_img, size):
        th = size
        tw = size
        
        h, w = pre_img.shape[1:3]

        if w == tw and h == tw:
            return pre_img, post_img

        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
    
        pre_img_cropped = pre_img[:, top : (top + th), left : (left + tw)]
        post_img_cropped = post_img[:, top : (top + th), left : (left + tw)]
        
        return (pre_img_cropped, post_img_cropped)

    def __getitem__(self, idx):
        current_pair = self.img_pairs[idx]
        I1 = current_pair[0]
        I2 = current_pair[1]
        
        # I1 = self.get_img(img_name_1)
        # I2 = self.get_img(img_name_2)

        if self.normalize:
            I1 = self.clip_1_99_percentile(I1)
            I2 = self.clip_1_99_percentile(I2)
            
            # I1 = (I1 - I1.mean()) / I1.std()
            # I2 = (I2 - I2.mean()) / I2.std()
            # I1 = (I1 - I1.min()) / (I1.max() - I1.min())
            # I2 = (I2 - I2.min()) / (I2.max() - I2.min())
            
        (I1, I2) = self.random_crop(I1, I2, self.patch_side)
        
        if self.transform:
            I1, I2 = self.transform(I1, I2)
        
        return (I1, I2)
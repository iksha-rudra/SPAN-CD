import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import ceil
import os
import numpy as np

from torch.utils.data import Dataset

class S2MTCP_dataset(Dataset):
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
        self.n_imgs = len(self.names)
        
        # load images
        self.imgs_1 = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        # print(f'Dataset: {self.names}')
        for i in range(0,len(self.names),2):
            im_name1 = self.names[i].strip()
            im_name2 = self.names[i+1].strip()

            # load and store each image
            print(f'reading....{im_name1}')
            print(f'reading....{im_name2}')

            I1 = self.get_s2mtcp_image(os.path.join(path, 'data_S21C', im_name1))
           
            s = I1.shape

            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            
            # generate path coordinates
            patches_per_image = 0
            for i in range(n1):
                for j in range(n2):

                    current_patch_coords = (im_name1,  
                                    [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                    [self.stride*(i + 1), self.stride*(j + 1)])
                    # print(current_patch_coords)
                    self.patch_coords.append(current_patch_coords)
                    patches_per_image += 1
            
            self.n_patches_per_image[im_name1] = patches_per_image
            
            self.n_patches += patches_per_image
            
            del I1
        
    def get_s2mtcp_image(self, img_path):
        image = torch.from_numpy(np.load(img_path)).float()
        image = torch.einsum('ijk->kij',image)
        image = image[0:13,:,:]
        return image

    def get_img(self, im_name):
        return self.get_s2mtcp_image(os.path.join(self.path, 'data_S21C', im_name))

    def __len__(self):
        return self.n_patches

    # def image_normalize_quantile(self,image, first=0.01, last=0.99):
    #     q = torch.tensor([first, last]).float()
    #     (C,M,N) = image.shape
    #     for i in range(C):
    #         qtnl = torch.quantile(torch.flatten(image[i,:,:]), q, dim=0)
    #         min = qtnl[0]
    #         max = qtnl[1]
    #         image[i,:,:] = (image[i,:,:] - min )/( max - min )
    #         res = image[i,:,:].clone()
    #         res[image[i,:,:] < 0] = 0.0
    #         res[image[i,:,:] > 1] = 1.0
    #         image[i,:,:] = res
    #     return image
    
    def image_normalize_quantile(self,image, first=0.01, last=0.99):
        
        q = torch.tensor([first, last]).float()
        (C,M,N) = image.shape
        for i in range(C):
            qtnl = torch.quantile(torch.flatten(image[i,:,:]), q, dim=0)
            min = qtnl[0]
            max = qtnl[1]
            image[i,:,:] = (image[i,:,:] - min )/( max - min )
            res = image[i,:,:].clone()
            res[image[i,:,:] < 0] = 0.0
            res[image[i,:,:] > 1] = 1.0
            image[i,:,:] = res
        return image

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        
        I = self.get_img(im_name)
        I1 = I[:, limits[0]:limits[1], limits[2]:limits[3]]
        del I
        
        I = self.get_img(im_name.replace('a','b'))
        I2 = I[:, limits[0]:limits[1], limits[2]:limits[3]]
        del I

        if self.normalize:
            I1 = self.image_normalize_quantile(I1)
            I2 = self.image_normalize_quantile(I2)
            # I1 = (I1 - I1.mean())/ (I1.std())
            # I2 = (I2 - I2.mean())/ (I2.std())
        
        return (I1, I2)
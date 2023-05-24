from torch.utils.data import Dataset
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.image as mpimg
import os
import numpy as np
from tifffile import imread, imwrite

class OSCD_SR_Dataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, 
                 path, 
                 fname = 'all.txt',
                 bands = ['B01','B02','B03','B04','B05',
                          'B06','B07','B08','B8A','B09','B11','B12'],
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
        self.transform = transform
        self.normalize = normalize
        self.path = path
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride
        
        with open(os.path.join(path, fname)) as f:
            lines = f.readline()
            
        names = lines.split(',')
        self.names = names
        self.n_imgs = len(self.names)
        
        n_pix = 0
        true_pix = 0
        
        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        # print(f'Dataset: {self.names}')
        for im_name in self.names:
            im_name = im_name.strip()
            # load and store each image
            print(f'reading....{im_name}')

            I1, I2, cm = self.read_sentinel_img_trio(self.path + im_name,
                                                band_list=bands,
                                                patch_size=patch_side,
                                                stride=stride)
            
            # image will be normalized
            if self.normalize:
                # I1 = self.image_normalize_quantile(I1)
                # I2 = self.image_normalize_quantile(I2)
                I1 = (I1 - I1.min()) / (I1.max() - I1.min())
                I2 = (I2 - I2.min()) / (I2.max() - I2.min())
                cm = (cm - cm.min()) / (cm.max() - cm.min())

        
            self.imgs_1[im_name] = I1
            self.imgs_2[im_name] = I2
            self.change_maps[im_name] = cm
            
            s = cm.shape

            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            
            # generate path coordinates
            patches_per_image = 0
            for i in range(n1):
                for j in range(n2):

                    current_patch_coords = (im_name,  
                                    [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                    [self.stride*(i + 1), self.stride*(j + 1)])
                    # print(current_patch_coords)
                    # check is it worth
                    limits = current_patch_coords[1]
                    label = self.change_maps[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
                    val, counts = np.unique(label, return_counts=True)

                    if (1 - (counts[0]/counts.sum())) > 0.00:  #At least 5% useful area with labels that are not 0
                        self.patch_coords.append(current_patch_coords)
                        patches_per_image += 1
                        n_pix += (counts[0] + counts[1])
                        true_pix += counts[0]
            
            self.n_patches_per_image[im_name] = patches_per_image
            self.n_patches += patches_per_image
                    
        self.weights = [ 10 * 2 *true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def reshape_for_torch(self, I):
        """Transpose image for PyTorch coordinates."""
        out = I.transpose((2, 0, 1))
        return torch.tensor(out.astype(np.float32)).double()

    def get_padded_image(self, image, 
                        patch_size=256,
                        stride=128):
    
        image = image.permute(1, 2, 0).numpy()
        s = image.shape
        x = s[0]
        y = s[1]
        
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
        out = self.reshape_for_torch(image)
        
        return out

    def read_sentinel_img(self, path, band_list):
        ''' For reading sentinel image'''
        I = None
    
        if len(band_list) > 0:
            band = band_list[0]
            
            I = imread(os.path.join(path, band + ".tif"))
            I = np.expand_dims(I,axis=2)
            
            for i in range(1, len(band_list)):
                band = band_list[i]
                b = imread(os.path.join(path, band + ".tif"))
                b = np.expand_dims(b,axis=2)
                I = np.concatenate((I,b),axis=2).astype(np.float32)
            
        return self.reshape_for_torch(I)

    def read_sentinel_img_trio(self, path, 
                            band_list = ['B01', 'B02','B03','B04','B05',
                                        'B06','B07','B08','B8A','B09',
                                        'B10','B11','B12'],
                            patch_size=256,
                            stride = 128):
        """Read cropped Sentinel-2 image pair and change map."""

        I1 = self.read_sentinel_img(path + '/imgs_1_ss/', band_list)
        I2 = self.read_sentinel_img(path + '/imgs_2_ss/', band_list)
        # I1 = self.read_sentinel_img(path + '/imgs_1_rect/', band_list)
        # I2 = self.read_sentinel_img(path + '/imgs_2_rect/', band_list)

        name = path.split('/')[-1]
        cm = mpimg.imread(path + '/cm/'+name+'-cm.tif')
        cm = cm - 1
        cm = np.expand_dims(cm,axis=2)
        
        cm = self.reshape_for_torch(cm)
        
        # I1 = self.get_padded_image(I1,patch_size,stride)
        # I2 = self.get_padded_image(I2,patch_size,stride)
        # cm = self.get_padded_image(cm,patch_size,stride)
        
        return I1, I2, cm

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def image_normalize_quantile(self,image, first=0.01, last=0.99):
        q = torch.tensor([first, last]).float()
        (C,M,N) = image.shape
        for i in range(C):
            qtnl = torch.quantile(image[i,:,:].view(-1), q, dim=0)
            min = qtnl[0]
            max = qtnl[1]
            image[i,:,:] = (image[i,:,:] - min )/( max - min )
            res = image[i,:,:].clone()
            res[image[i,:,:] < 0] = 0.0
            res[image[i,:,:] > 1] = 1.0
            # res = res * 255
            image[i,:,:] = res
        return image

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        
        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        label = self.change_maps[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        
        sample = {'I1': I1, 'I2': I2, 'label': label, 'fname': im_name}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
        
    def get_weights(self):
        return self.weights
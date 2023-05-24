'''
Currently working

RandomFlip
RandomRotation
GaussianBlur
ToGray
RandomResizedCrop
Normalize
RandomNoise
RandomGeomRotation
Solarization
BandSwap
BandTranslation

'''

import torch
import torch.nn as nn
import torchvision.transforms as tr
import torch.optim as optim
import random
import numpy as np
import cv2

from torchvision import transforms
from transforms import transforms
from display import PlotImagesAndMask

class RandomFlip(object):
    """Flip randomly the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, sample):

        I1 = sample['I1']
        I2 = sample['I2']
        label = sample['label']
        fname = sample['fname']
        lbel_list = sample['list']
        
        if random.random() <= self.p:
            I1 =  I1.numpy()[:,:,::-1].copy()
            I1 = torch.from_numpy(I1)
            
            I2 =  I2.numpy()[:,:,::-1].copy()
            I2 = torch.from_numpy(I2)
            
            label =  label.numpy()[:,:,::-1].copy()
            label = torch.from_numpy(label)
            
            lbl_lst = []
            for lbl in lbel_list:
                lbl =  lbl.numpy()[:,:,::-1].copy()
                lbl = torch.from_numpy(lbl)
                lbl_lst.append(lbl)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname, 'list': lbl_lst}

class RandomRotation(object):
    """Rotate randomly the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, sample):
        I1 = sample['I1']
        I2 = sample['I2']
        label = sample['label']
        fname = sample['fname']
        label_list = sample['list']
        
        if random.random() <= self.p:
            n = random.randint(1, 4)
            if n:
                I1 =  sample['I1'].numpy()
                I1 = np.rot90(I1, n, axes=(1, 2)).copy()
                I1 = torch.from_numpy(I1)
                
                I2 =  sample['I2'].numpy()
                I2 = np.rot90(I2, n, axes=(1, 2)).copy()
                I2 = torch.from_numpy(I2)
                
                label =  sample['label'].numpy()
                label = np.rot90(label, n, axes=(1, 2)).copy()
                label = torch.from_numpy(label)
                
                lbl_lst = []
                for lbl in label_list:
                    lbl =  lbl.numpy()
                    lbl = np.rot90(lbl, n, axes=(1, 2)).copy()
                    lbl = torch.from_numpy(lbl)
                    lbl_lst.append(lbl)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname, 'list': lbl_lst}
    
class GaussianBlur(object):
    """Apply Gaussian Blur the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:
            plotimgs = PlotImagesAndMask()
            I1 = plotimgs.change_to_numpy(I1)
            I2 = plotimgs.change_to_numpy(I2)
            label = plotimgs.change_to_numpy(label)
            
            I1, I2, label = transforms.Compose([
            transforms.GaussianBlur(kernel_size=9)
            ])(I1, I2, label)

            I1 = plotimgs.change_to_tensor(I1)
            I2 = plotimgs.change_to_tensor(I2)
            label = plotimgs.change_to_tensor(label)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}
    
class ElasticTransform(object):
    """Apply Gaussian Blur the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:
            plotimgs = PlotImagesAndMask()
            I1 = plotimgs.change_to_numpy(I1)
            I2 = plotimgs.change_to_numpy(I2)
            label = plotimgs.change_to_numpy(label)
            
            I1, I2, label = transforms.Compose([
            transforms.ElasticTransform()
            ])(I1, I2, label)
            
            label = np.expand_dims(label, axis=2)

            I1 = plotimgs.change_to_tensor(I1)
            I2 = plotimgs.change_to_tensor(I2)
            label = plotimgs.change_to_tensor(label)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}
    
class ToGray(object):
    """Apply gray scaling the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:
            plotimgs = PlotImagesAndMask()
            I1 = plotimgs.change_to_numpy(I1)
            I2 = plotimgs.change_to_numpy(I2)
            label = plotimgs.change_to_numpy(label)
            
            I1, I2, label = transforms.Compose([
            transforms.ToGray(13)
            ])(I1, I2, label)

            I1 = plotimgs.change_to_tensor(I1)
            I2 = plotimgs.change_to_tensor(I2)
            label = plotimgs.change_to_tensor(label)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}
    
class RandomResizedCrop(object):
    """Apply random crop the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:
            plotimgs = PlotImagesAndMask()
            I1 = plotimgs.change_to_numpy(I1)
            I2 = plotimgs.change_to_numpy(I2)
            label = plotimgs.change_to_numpy(label)
            
            I1, I2, label = transforms.Compose([
            transforms.RandomResizedCrop(crop_size=100, 
                                            target_size=256)
            ])(I1, I2, label)

            I1 = plotimgs.change_to_tensor(I1)
            I2 = plotimgs.change_to_tensor(I2)
            label = plotimgs.change_to_tensor(label)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}    
    
class StdNormalize(object):
    """Apply normalize the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:
        
            pre_image_mean = torch.mean(I1, dim = [1,2])
            pre_image_std = torch.std(I1,dim=(1,2))
            
            post_image_mean = torch.mean(I2,dim=(1,2))
            post_image_std = torch.std(I2,dim=(1,2))
            
            I1, I2, label = transforms.Compose([
            transforms.Normalize(pre_img_mean=pre_image_mean,
                                    post_img_mean=post_image_mean,
                                    pre_img_std=pre_image_std,
                                    post_img_std=post_image_std)
            ])(I1, I2, label)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}   
    
class Normalize(object):
    """Apply normalize the images in a sample."""
    def __init__(self, type='minmax', p=0.5) -> None:
        self.p = p
        self.type = type
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:
        
            (C,M,N) = I1.shape
            
            if self.type == 'minmax':
                for i in range(C):
                    I1[i,:,:] = (I1[i,:,:] - I1[i,:,:].min())/(I1[i,:,:].max()-I1[i,:,:].min())
                    I2[i,:,:] = (I2[i,:,:] - I2[i,:,:].min())/(I2[i,:,:].max()-I2[i,:,:].min())

            elif self.type == 'std':
                for i in range(C):
                    I1[i,:,:] = (I1[i,:,:] - I1[i,:,:].mean())/(I1[i,:,:].std())
                    I2[i,:,:] = (I2[i,:,:] - I2[i,:,:].mean())/(I2[i,:,:].std())

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}       
    
class RandomNoise(object):
    """Apply Random noise the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:            
            (C,M,N) = I1.shape
            
            x = torch.zeros(C, M, N, dtype=torch.float16)
            x = x + (0.1**0.5)*torch.randn(C, M, N)
            I1 = I1 + x
        
            x = torch.zeros(C, M, N, dtype=torch.float16)
            x = x + (0.1**0.5)*torch.randn(C, M, N)
            I2 = I2 + x        

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}

class RandomBrightness(object):
    """Apply random brightness the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
        
    def adjust_brightness(img, value=0):
        return img
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:
            plotimgs = PlotImagesAndMask()
            I1 = plotimgs.change_to_numpy(I1)
            I2 = plotimgs.change_to_numpy(I2)
            label = plotimgs.change_to_numpy(label)
            
            I1 = self.adjust_brightness(I1)
            I2 = self.adjust_brightness(I2)

            I1 = plotimgs.change_to_tensor(I1)
            I2 = plotimgs.change_to_tensor(I2)
            label = plotimgs.change_to_tensor(label)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}  
    
class RandomShift(object):
    """Apply random shift the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:
            plotimgs = PlotImagesAndMask()
            I1 = plotimgs.change_to_numpy(I1)
            I2 = plotimgs.change_to_numpy(I2)
            label = plotimgs.change_to_numpy(label)
            
            I1, I2, label = transforms.Compose([
            transforms.RandomShift()
            ])(I1, I2, label)

            I1 = plotimgs.change_to_tensor(I1)
            I2 = plotimgs.change_to_tensor(I2)
            label = plotimgs.change_to_tensor(label)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}
    
class RandomGeomRotation(object):
    """Apply random shift the images in a sample."""
    def __init__(self, degree=45, p=0.5) -> None:
        self.p = p
        self.degrees = degree
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:            
            plotimgs = PlotImagesAndMask()
            I1 = plotimgs.change_to_numpy(I1)
            I2 = plotimgs.change_to_numpy(I2)
            label = plotimgs.change_to_numpy(label)
            
            I1, I2, label = transforms.Compose([
            transforms.RandomRotation(degrees=self.degrees)
            ])(I1, I2, label)
            
            label = np.expand_dims(label, axis=2)

            I1 = plotimgs.change_to_tensor(I1)
            I2 = plotimgs.change_to_tensor(I2)
            label = plotimgs.change_to_tensor(label)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}  
    
class Solarization(object):
    """Apply Solarization the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def Solarize(self, image):

        (C,M,N) = image.shape
        for i in range(C):
            channel = image[i,:,:]
            max_pixel = torch.max(channel)
            threshold = max_pixel / 2
            channel[channel < threshold] = max_pixel - channel[channel < threshold]
            image[i,:,:] = channel
        return image
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']

        if random.random() <= self.p:
            I1 = self.Solarize(I1)
            I2 = self.Solarize(I2)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}
    
class ColorJitter(object):
    """Apply color jitter the images in a sample."""
    
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def jitter(self, img):
        img = img.to(torch.float64)
        # Define ColorJitter transform
        color_jitter = tr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)

        # Apply transform to image
        I_rgb = torch.cat([img[4,:,:], img[3,:,:], img[2,:,:]], dim=0)
        I_rgb = I_rgb / torch.finfo(torch.float64).max
        I_rgb = color_jitter(I_rgb)
        I_rgb = I_rgb * torch.finfo(torch.float64).max
        
        img[4,:,:] = I_rgb[0,:,:]
        img[3,:,:] = I_rgb[1,:,:]
        img[2,:,:] = I_rgb[2,:,:]
        
        return img
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:
            I1 = self.jitter(I1)
            I2 = self.jitter(I2)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}
    
class BandSwap(object):
    """Some Information about MyModule"""
    def __init__(self, num_swaps=3, p = 0.5):
        self.num_swaps = num_swaps
        self.p = p

    def random_band_swap(self, image):
        
        num_bands = image.shape[0]
        band_pairs = [(i, j) for i in range(num_bands) for j in range(i+1, num_bands)]
        swap_indices = random.sample(band_pairs, self.num_swaps)
        swapped_image = image.clone()

        for i, j in swap_indices:
            swapped_image[[i, j], :, :] = swapped_image[[j, i], :, :]
        
        return swapped_image
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:
            I1 = self.random_band_swap(I1)
            I2 = self.random_band_swap(I2)

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}

class BandTranslation(object):
    """Apply gray scaling the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() <= self.p:
            (C,M,N) = I1.shape

            bias = torch.empty(C, M, N).uniform_(-0.1, 0.1)

            I1 = I1 + bias
            I2 = I2 + bias

        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}
    
class PCA_Fancy(object):
    
    def __init__(self, p=0.5) -> None:
        self.p = p

    def pca_color_augmentation(img, eigenvalues, eigenvectors, mean):
        """Applies PCA color augmentation to a multi-spectral image."""
        img = np.asarray(img, dtype=np.float32)  # convert to numpy array
        img = img / 255.0  # scale pixel values to be between 0 and 1

        # get number of bands
        num_bands = img.shape[-1]

        # apply PCA color augmentation to each band separately
        for band_idx in range(num_bands):
            # flatten band into 1D array
            flat_band = img[..., band_idx].flatten()

            # subtract the mean from the band values
            flat_band = flat_band - mean[band_idx]

            # generate random coefficients for each eigenvector
            coeffs = np.random.normal(scale=0.1, size=3)

            # transform the band values using the eigenvectors and coefficients
            pca_transform = eigenvectors * np.sqrt(eigenvalues) * coeffs
            pca_transformed = np.dot(flat_band, pca_transform)

            # add the mean back to the color-transformed band
            pca_augmented_band = pca_transformed + mean[band_idx]

            # clip pixel values to be within the range of 0-255
            pca_augmented_band = np.clip(pca_augmented_band, 0.0, 255.0)

            # reshape the augmented band back to its original shape
            pca_augmented_band = pca_augmented_band.reshape(img.shape[:-1])

            # replace the original band with the augmented band
            img[..., band_idx] = pca_augmented_band

        # convert the augmented image back to uint8 format
        pca_augmented = np.asarray(img, dtype=np.uint8)

        return pca_augmented
    
    def __call__(self, sample):
        I1, I2, label, fname = sample['I1'], sample['I2'], sample['label'], sample['fname']
        
        if random.random() >= self.p:
                    # load image data
            img_data = np.load('image_data.npy')

            # compute mean and covariance
            mean = np.mean(img_data, axis=(0, 1))
            cov = np.cov(img_data.reshape(-1, img_data.shape[-1]).T)

            # compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(cov)

            # apply PCA color augmentation to an image
            img = img_data[0]  # choose an example image to augment
            augmented_img = self.pca_color_augmentation(img, eigenvalues, eigenvectors, mean)
            
        return {'I1': I1, 'I2': I2, 'label': label, 'fname': fname}


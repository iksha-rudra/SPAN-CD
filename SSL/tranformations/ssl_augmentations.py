

import torch
import torch.nn as nn
import torchvision.transforms as tr
import torch.optim as optim
import random
import numpy as np
import cv2

from tranformations import ssl_transforms

'''
Normalise
RandomFlip
RandomRotation
RandomResizedCrop
RandomNoise
RandomChannelDrop
BandTranslation
BandSwap
RandomPixelRemove
'''

#Spatial Transoforms


class RandomFlip(object):
    """Flip randomly the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, image):
        
        if random.random() <= self.p:
            image =  image.numpy()[:,:,::-1].copy()
            image = torch.from_numpy(image)

        return image

class RandomRotation(object):
    """Rotate randomly the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, image):
        
        if random.random() <= self.p:
            n = random.randint(1, 4)
            if n:
                image =  image.numpy()
                image = np.rot90(image, n, axes=(1, 2)).copy()
                image = torch.from_numpy(image)

        return image
    
class RandomResizedCrop(object):
    """Apply random crop the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, image):
    
        if random.random() <= self.p:
            image = image.numpy()
            image = np.einsum('ijk->jki',image)

            image = ssl_transforms.Compose([
            ssl_transforms.RandomResizedCrop(crop_size=224, 
                                            target_size=224)
            ])(image)

            image = torch.from_numpy(image)
            image = torch.einsum('ijk->kij',image)

        return image 

class RandomNoise(object):
    """Apply random crop the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, image):
    
        if random.random() <= self.p:
            image = image.numpy()
            image = np.einsum('ijk->jki',image)

            image = ssl_transforms.Compose([
            ssl_transforms.RandomNoise()
            ])(image)

            image = torch.from_numpy(image)
            image = torch.einsum('ijk->kij',image)

        return image   

class RandomChannelDrop(object):
    """ Random Channel Drop """
    
    def __init__(self, min_n_drop=1, max_n_drop=2, p = 0.5):
        self.min_n_drop = min_n_drop
        self.max_n_drop = max_n_drop
        self.p = p

    def __call__(self, sample):
        
        if random.random() <= self.p:

            n_channels = random.randint(self.min_n_drop, self.max_n_drop)
            channels = np.random.choice(range(sample.shape[0]), size=n_channels, replace=False)

            for c in channels:
                sample[c, :, :] = 0   
 
        return sample
    
class BandSwap(object):
    """Some Information about MyModule"""
    def __init__(self, num_swaps=2, p = 0.5):
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
        
        if random.random() <= self.p:
            sample = self.random_band_swap(sample)

        return sample
    
    
    
    
    
    

class BandTranslation(object):
    """Apply gray scaling the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, sample):
        
        if random.random() <= self.p:
            (C,M,N) = sample.shape

            bias = torch.empty(C, M, N).uniform_(-0.1, 0.1)

            sample = sample + bias

        return sample

# from review paper
class RandomBrightness(object):
    """ Random Brightness """
    
    def __init__(self, brightness=0.2, p = 0.5):
        self.brightness = brightness
        self.p = p

    def __call__(self, sample):
        if random.random() <= self.p:

            sample = sample.numpy()
            sample = np.einsum('ijk->jki',sample)
            
            s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            img = sample * s
        
            sample =  img.astype(np.uint8)
            
            sample = torch.from_numpy(sample)
            sample = torch.einsum('ijk->kij',sample)
            
        return sample
    
class RandomContrast(object):
    """ Random Contrast """
    
    def __init__(self, contrast=0.2, p = 0.5):
        self.contrast = contrast
        self.p = p

    def __call__(self, sample):
        
        if random.random() <= self.p:
            
            sample = sample.numpy()
            sample = np.einsum('ijk->jki',sample)
            
            s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            mean = np.mean(sample, axis=(0, 1))
            
            sample = ((sample - mean) * s + mean).astype(np.uint8)
            
            sample = torch.from_numpy(sample)
            sample = torch.einsum('ijk->kij',sample)
            
        return sample

#Spectral Transformations
class RandomPixelRemove(object):
    """Apply Random noise the images in a sample."""
    def __init__(self, percent=0.2, p=0.5) -> None:
        self.p = p
        self.percent = percent
    
    def __call__(self, image):
        
        if random.random() <= self.p:            
            (C,M,N) = image.shape

            for i in range(3000):
                x = int(255 * random.random())
                y = int(255 * random.random())

                image[:,x,y] = 0.0

        return image

class GaussianBlur(object):
    """Apply Gaussian Blur the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, image):
        
        if random.random() <= self.p:
            image = image.numpy()
            image = np.einsum('ijk->jki',image)
            
            image = ssl_transforms.Compose([
            ssl_transforms.GaussianBlur(kernel_size=5)
            ])(image)
            
            image = torch.from_numpy(image)
            image = torch.einsum('ijk->kij',image)

        return image
    
class ToGray(object):
    """Apply gray scaling the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, image):
    
        if random.random() <= self.p:
            image = image.numpy()
            image = np.einsum('ijk->jki',image)
            
            image = ssl_transforms.Compose([
            ssl_transforms.ToGray(13)
            ])(image)
            
            image = torch.from_numpy(image)
            image = torch.einsum('ijk->kij',image)

        return image
    
class Normalize(object):
    """Apply normalize the images in a sample."""
    def __init__(self, type='minmax', p=0.5) -> None:
        self.p = p
        self.type = type
    
    def __call__(self, image):
        
        if random.random() <= self.p:
        
            (C,M,N) = image.shape
            
            if self.type == 'minmax':
                for i in range(C):
                    image[i,:,:] = (image[i,:,:] - image[i,:,:].min())/ \
                        (image[i,:,:].max()-image[i,:,:].min())

            elif self.type == 'std':
                for i in range(C):
                    image[i,:,:] = (image[i,:,:] - image[i,:,:].mean())/(image[i,:,:].std())

        return image    
    
class RandomShift(object):
    """Apply random shift the images in a sample."""
    def __init__(self, p=0.5) -> None:
        self.p = p
    
    def __call__(self, image):
        
        if random.random() <= self.p:
            
            image = image.numpy()
            image = np.einsum('ijk->jki',image)

            image = ssl_transforms.Compose([
            ssl_transforms.RandomShift()
            ])(image)

            image = torch.from_numpy(image)
            image = torch.einsum('ijk->kij',image)

        return image 
    
class RandomGeomRotation(object):
    """Apply random shift the images in a sample."""
    def __init__(self, degree=45, p=0.5) -> None:
        self.p = p
        self.degrees = degree
    
    def __call__(self, image):
        
        if random.random() <= self.p:            
            image = image.numpy()
            image = np.einsum('ijk->jki',image)
            
            image = ssl_transforms.Compose([
            ssl_transforms.RandomRotation(degrees=self.degrees)
            ])(image)
            
            image = torch.from_numpy(image)
            image = torch.einsum('ijk->kij',image)

        return image
    
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
        
        if random.random() <= self.p:
            sample = self.Solarize(sample)

        return sample
    
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
        
        if random.random() <= self.p:
            sample = self.jitter(sample)

        return sample
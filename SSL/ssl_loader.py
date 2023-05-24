import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from cvtorchvision import cvtransforms
from torchvision import transforms as tr
import numpy as np
from tranformations.ssl_augmentations import    RandomFlip, \
                                                RandomRotation, \
                                                RandomResizedCrop, \
                                                RandomNoise, \
                                                RandomChannelDrop, \
                                                BandTranslation, \
                                                BandSwap, \
                                                RandomBrightness, \
                                                RandomContrast, \
                                                RandomPixelRemove, \
                                                GaussianBlur, \
                                                ToGray, \
                                                Normalize

class TransformSSL(nn.Module):
    def __init__(self):
        self.transform = tr.Compose([
                RandomResizedCrop(p = 1.0),
                RandomFlip(p = 0.75),
                RandomNoise(p = 0.75),
                RandomChannelDrop(p = 0.75),
                BandSwap(p = 0.75)
        ])
        self.transform_prime = tr.Compose([
                RandomResizedCrop(p = 1.0),
                RandomFlip(p = 0.75),
                RandomNoise(p = 0.75),
                RandomChannelDrop(p = 0.75),
                BandSwap(p = 0.75)
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
    
 
# class TransformSSL_2(nn.Module):
#     def __init__(self):
#         self.transform = tr.Compose([
#             RandomRotation(p=0.5),
#             RandomFlip(p=0.5),
#         ])
#         self.transform_prime = tr.Compose([
#             RandomRotation(p=0.5),
#             RandomFlip(p=0.5),
#         ])

#     def __call__(self, x, y):
#         y1 = self.transform(x)
#         y2 = self.transform_prime(y)
#         return y1, y2  

# class TransformSSL_3(nn.Module):
#     def __init__(self):
#         self.transform = tr.Compose([
#             RandomFlip(p=0.75),
#             RandomNoise(p=0.75)
#         ])
#         self.transform_prime = tr.Compose([
#             RandomFlip(p=0.75),
#             RandomNoise(p=0.75)
#         ])

#     def __call__(self, x, y):
#         y1 = self.transform(x)
#         y2 = self.transform_prime(y)
#         return y1, y2 
        
# class TransformSSL_2(nn.Module):
#     def __init__(self):
#         self.transform = tr.Compose([
#             ToGray(p=0.5),
#             GaussianBlur(p=0.5),
#             RandomRotation(p=0.5),
#             RandomResizedCrop(p=1.0),
#             # RandomNoise(p=0.2),
#             RandomChannelDrop(p=0.2),
#             # BandTranslation(p=0.2),
#             # BandSwap(p=0.2),
#             # RandomPixelRemove(p=0.2)
#         ])
#         self.transform_prime = tr.Compose([
#             ToGray(p=0.5),
#             GaussianBlur(p=0.5),
#             RandomRotation(p=0.5),
#             RandomResizedCrop(p=1.0),
#             # RandomNoise(p=0.8),
#             RandomChannelDrop(p=0.5),
#             # BandTranslation(p=0.5),
#             # BandSwap(p=0.5),
#             # RandomPixelRemove(p=0.5)
#         ])

#     def __call__(self, x, y):
#         y1 = self.transform(x)
#         y2 = self.transform_prime(y)
#         return y1, y2
    
    
from dataset.s2mtcp_dataset import S2MTCP_dataset
from dataset.oscd_dataset_ssl import OSCD_SSL
from dataset.sentinel2_oscd_ssl import Sentinel2_OSCD_SSL
from dataset.sentinel2_s2mtcp_ssl import Sentinel2_S2MTCP_SSL
from dataset.sentinel2_sen12ms_ssl import Sentinel2_SEN12MS_SSL
from torch.utils.data import DataLoader
    
def get_s2mtcp_ssl_loader(path,
                          bands, 
                        batch_size,
                        patch_side, 
                        stride):
    
    # transform = TransformSSL_2()
    transform = TransformSSL()
    
    train_dataset = Sentinel2_S2MTCP_SSL(path=path,
                                   transform=transform,
                                   patch_side=patch_side,
                                   stride=patch_side,
                                   bands=bands,
                                   normalize=True)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last=False)
    return train_loader 

def get_oscd_ssl_loader(path, 
                        batch_size,
                        patch_side, 
                        stride):
    
    transform = TransformSSL()
    
    train_dataset = Sentinel2_OSCD_SSL(path=path,
                            patch_side=patch_side, 
                            stride = stride,
                             transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last=True)
    return train_loader 

def get_sen12ms_ssl_loader(path, 
                        batch_size,
                        patch_side, 
                        stride):
    
    transform = TransformSSL()
    
    train_dataset = Sentinel2_SEN12MS_SSL(path=path,
                            patch_side=patch_side, 
                            stride = stride,
                             transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last=True)
    return train_loader 
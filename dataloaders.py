import torch
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from torch.utils.data import ConcatDataset

from dataset.oscd_sr_dataset import OSCD_SR_Dataset
from dataset.oscd_dataset import OSCD_Dataset

from augmentations import   RandomFlip, \
                            RandomRotation, \
                            GaussianBlur, \
                            ToGray, \
                            RandomResizedCrop, \
                            Normalize, \
                            RandomNoise, \
                            RandomGeomRotation, \
                            Solarization, \
                            BandSwap, \
                            BandTranslation

# def get_transform(data_aug):
#     data_transform = None
#     if data_aug:
#         data_transform = tr.Compose([
#             RandomResizedCrop(p=0.5),
#             RandomRotation(p=0.5),
#             RandomFlip(p=0.5),
#             tr.RandomApply(
#                 [GaussianBlur(p=0.5),
#                  ToGray(p=0.5),
#                  RandomNoise(p=0.5),
#                  RandomGeomRotation(0.5)
#                  ],
#                 p=0.8
#             ),
#             Solarization(p=0.5),
#             BandSwap(p=0.5),
#             BandTranslation(p=0.5),
#             StdNormalize(p=1.0)
#         ])
#     else:
#         data_transform = None
#     return data_transform

def get_augmented_dataset(path,
                    fname,
                    patch_side,
                    stride,
                    bands):
        
    transform_rotate_flip = tr.Compose([
        RandomFlip(p=1.0),
        RandomRotation(p=1.0)
    ])
    
    transform_gaussian_blur = tr.Compose([
        GaussianBlur(p=1.0),
        RandomFlip(p=1.0),
        RandomRotation(p=1.0)
    ])
    
    transform_bandswap = tr.Compose([
        BandSwap(p=1.0),
        RandomFlip(p=1.0),
        RandomRotation(p=1.0)
    ])
    
    transform_translation = tr.Compose([
        BandTranslation(p=1.0),
        RandomFlip(p=1.0),
        RandomRotation(p=1.0)
    ])
    
    transform_solarisation = tr.Compose([
        Solarization(p=1.0),
        RandomFlip(p=1.0),
        RandomRotation(p=1.0)
    ])
    
    transform_resize_crop = tr.Compose([
        RandomResizedCrop(p=1.0),
        RandomFlip(p=1.0),
        RandomRotation(p=1.0)
    ])
    
    transform_to_gray = tr.Compose([
        ToGray(p=1.0),
        RandomFlip(p=1.0),
        RandomRotation(p=1.0)
    ])
    
    # create different versions of the dataset with different transformations
        
    # dataset_rotate_flip = OSCD_SR_Dataset(path=path,
    #                                     fname=fname,
    #                                     patch_side=patch_side,
    #                                     stride=stride,
    #                                     normalize=True,
    #                                     transform=transform_rotate_flip,
    #                                     bands=bands
    #                                     )
    
    dataset_rotate_flip = OSCD_Dataset(path=path,
                                    fname=fname,
                                    patch_side=patch_side,
                                    stride=stride,
                                    normalize=True,
                                    transform=transform_rotate_flip,
                                    bands=bands
                                    )
    
    # dataset_gaussian_blur = ChangeDetectionDataset(path,
    #                                 fname,
    #                                 bands,
    #                                 patch_side,
    #                                 stride,
    #                                 True,
    #                                 transform_gaussian_blur
    #                                 )

    # concatenate the datasets
    # augmented_dataset = ConcatDataset([dataset_rotate_flip])

    augmented_dataset = dataset_rotate_flip

    
    return augmented_dataset, dataset_rotate_flip.weights

def get_normalised_dataset(path,
                    fname,
                    patch_side,
                    stride,
                    bands):

    # normalised_dataset = OSCD_SR_Dataset(path=path,
    #                                 fname=fname,
    #                                 patch_side=patch_side,
    #                                 stride=stride,
    #                                 normalize=True,
    #                                 transform=None,
    #                                 bands=bands
    #                                 )
    
    normalised_dataset = OSCD_Dataset(path=path,
                                fname=fname,
                                patch_side=patch_side,
                                stride=stride,
                                normalize=True,
                                transform=None,
                                bands=bands
                                )
    
    print(patch_side, stride, fname, path)
    
    return normalised_dataset

def train_dataloader(path,
                     bands, 
                     batch_size, 
                     patch_side, 
                     stride, 
                     shuffle=True):
    
    dataset, weights = get_augmented_dataset(path=path,
                                fname = 'train.txt', 
                                patch_side = patch_side, 
                                stride = stride,
                                bands=bands)
    
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 0)
    return train_loader, weights 

def val_dataloader(path, 
                   bands,
                   batch_size, 
                   patch_side, 
                   stride, 
                   shuffle=True):
    
    dataset, weights = get_augmented_dataset(path=path,
                                fname = 'test.txt', 
                                patch_side = patch_side, 
                                stride = stride,
                                bands=bands)
    
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 0)
    return train_loader, dataset 

def test_dataloader(path,
                    bands, 
                    batch_size, 
                    patch_side, 
                    stride, 
                    shuffle=False):
    
    dataset = get_normalised_dataset(path=path,
                                fname = 'test.txt', 
                                patch_side = patch_side, 
                                stride = stride,
                                bands=bands)
    
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 0)
    return train_loader, dataset 
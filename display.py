import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2

class PlotImagesAndMask(nn.Module):
    """Some Information about PlotImageAndMask"""
    def __init__(self):
        super(PlotImagesAndMask, self).__init__()
        
    def normalize8(self, I):
        mn = I.min()
        mx = I.max()

        mx -= mn

        I = ((I - mn)/mx) * 255
        I = I.astype(np.uint8)
        return I
    
    def change_to_numpy(self, image):
        image = torch.permute(image,(1,2,0))
        np_arr = image.cpu().detach().numpy()
        return np_arr[::-1, :, :]
    
    def change_to_tensor(self, image):
        image = torch.from_numpy(image.copy())
        image = torch.permute(image,(2,0,1))
        return image
    
    def get_rgb(self, img):

        r = self.normalize8(img[:,:,4])
        g = self.normalize8(img[:,:,3])
        b = self.normalize8(img[:,:,2])

        rgb = np.stack([r,g,b])
        brg = rgb.transpose((1, 2, 0))

        return brg

    def forward(self, sample):
        
        label = sample['label']
        I1 = sample['I1']
        I2 = sample['I2']
        
        I1 = self.change_to_numpy(I1)
        I2 = self.change_to_numpy(I2)
        label = self.change_to_numpy(label)

        return self.get_rgb(I1), self.get_rgb(I2), label

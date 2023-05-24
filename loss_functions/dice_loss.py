import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import segmentation_models_pytorch as smp

class DiceLoss(nn.Module):
    """Some Information about DiceLoss"""
    def __init__(self, log_cosh=False):
        super(DiceLoss, self).__init__()
        self.log_cosh = log_cosh
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass',
                                             from_logits=True,
                                             smooth=1.0)

    def forward(self, output, target):
        
        lss = self.dice_loss(output, target)
    
        if self.log_cosh:
            lss = torch.log((torch.exp(lss) + torch.exp(-lss)) / 2.0)

        return lss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import segmentation_models_pytorch as smp

# class FocalLoss(nn.Module):
#     """Some Information about DiceLoss"""
#     def __init__(self):
#         super(FocalLoss, self).__init__()
#         self.focal_loss = smp.losses.FocalLoss(alpha=0.8,
#                                                gamma=2.0,
#                                             mode='multiclass')

#     def forward(self, output, target):
        
#         return self.focal_loss(output, target)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(input=inputs,
                                     target=targets,
                                     weight=self.weight, 
                                     reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss


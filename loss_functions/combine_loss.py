import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from loss_functions.edge_loss import EdgeLoss
from loss_functions.dice_loss import DiceLoss 
from loss_functions.bce_loss import BCELoss

class CombineLoss(nn.Module):
    """Some Information about CombineLoss"""
    def __init__(self, weights=None):
        super(CombineLoss, self).__init__()
        self.weights = weights

    def forward(self, prediction, target):
        """Calculating the loss"""
        loss = 0
        
        # EL = EdgeLoss(KSIZE=7)
        # el = EL(prediction, target)
        # loss += el
        
        DL = DiceLoss(log_cosh=True)
        dl = DL(prediction, target)
        loss += dl
        
        BL = BCELoss(weights=self.weights)
        fl = BL(prediction, target)
        loss += fl

        return loss

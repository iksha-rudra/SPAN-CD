import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from loss_functions.bce_loss import BCELoss
from loss_functions.dice_loss import DiceLoss 

class DS_Loss(nn.Module):
    """Some Information about DS_Loss"""
    def __init__(self, weights=None):
        super(DS_Loss, self).__init__()
        self.weights = weights
        self.BL = BCELoss(weights=self.weights)
        self.DL = DiceLoss(log_cosh=True)

    def forward(self, prediction, target):
        """Calculating the loss"""
        loss = 0
        
        dl = self.DL(prediction, target)
        loss += dl
    
        fl = self.BL(prediction, target)
        loss += fl

        return loss
    
class DS_Loss_all(nn.Module):
    """Some Information about DS_Loss_all"""
    def __init__(self, weights=None):
        super(DS_Loss_all, self).__init__()
        self.ds_loss = DS_Loss(weights=weights)

    def forward(self, prediction_list, target_list):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        label1 = Variable(torch.squeeze(target_list[0],dim=1).long()).to(device=device)
        label2 = Variable(torch.squeeze(target_list[1],dim=1).long()).to(device=device)
        label3 = Variable(torch.squeeze(target_list[2],dim=1).long()).to(device=device)
        label4 = Variable(torch.squeeze(target_list[3],dim=1).long()).to(device=device)
        label5 = Variable(torch.squeeze(target_list[4],dim=1).long()).to(device=device)
        label6 = Variable(torch.squeeze(target_list[5],dim=1).long()).to(device=device)
        
        loss = 0

        loss = loss + 1.0 * self.ds_loss(prediction_list[0], label1)
        # loss = loss + 0.16 * self.ds_loss(prediction_list[1], label2)
        # loss = loss + 0.16 * self.ds_loss(prediction_list[2], label3)
        # loss = loss + 0.16 * self.ds_loss(prediction_list[3], label4)
        # loss = loss + 0.16 * self.ds_loss(prediction_list[4], label5)
        # loss = loss + 0.16 * self.ds_loss(prediction_list[5], label6)

        return loss
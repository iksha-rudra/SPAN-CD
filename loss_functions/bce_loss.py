import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BCELoss(nn.Module):
    """Some Information about BCELoss"""
    def __init__(self, weights=None):
        super(BCELoss, self).__init__()
        self.nllloss = nn.NLLLoss(weight=weights)

    def forward(self, output, target):
        lsm = nn.LogSoftmax(dim=1)
        output = lsm(output)

        return self.nllloss(output, target)
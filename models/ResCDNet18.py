import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    def forward(self, x):
        return self.bn(self.c(x))

class ResCDNet18(nn.Module):
    """Some Information about ResCDNet18"""
    def __init__(self, in_channel, out_channel):  
        super(ResCDNet18, self).__init__()
        
        self.net = models.resnet18(weights='DEFAULT')
        self.net.fc = nn.Identity()
        # self.net.avgpool = nn.Identity()
        self.net.conv1 = torch.nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # self.relu = nn.ReLU()
        
        # self.decoder_11 = ConvBlock(768, 256, 3, 1, 1)
        # self.decoder_12 = ConvBlock(256, 256, 3, 1, 1)

        # self.decoder_21 = ConvBlock(384, 128, 3, 1, 1)
        # self.decoder_22 = ConvBlock(128, 128, 3, 1, 1)

        # self.decoder_31 = ConvBlock(192, 64, 3, 1, 1)
        # self.decoder_32 = ConvBlock(64, 64, 3, 1, 1)

        # self.decoder_41 = ConvBlock(128, 32, 3, 1, 1)
        # self.decoder_42 = ConvBlock(32, 32, 3, 1, 1)
        
        # self.decoder_51 = ConvBlock(32, 16, 3, 1, 1)
        # self.decoder_52 = ConvBlock(16, 16, 3, 1, 1)
        # self.decoder_53 = ConvBlock(16, out_channel, 3, 1, 1)
        
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        

    def forward(self, x1, x2):

        x1 = self.net(x1)
        x2 = self.net(x2)
        
        # print(x1.shape)
        
        # x = torch.abs(x1-x2)
        # x_out = torch.cat([x1, x2, x],dim=1)
        
        return x1
    
if __name__ == '__main__':
    model = ResCDNet18(13,2)
    print(model.children())
    x1 = torch.rand(1,13,256,256)
    x2 = torch.rand(1,13,256,256)
    out = model(x1,x2)
    print(out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models


class Decoder(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(Decoder, self).__init__()
        
        self.conv = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1) #keep ratio
        self.conv_trans = nn.ConvTranspose2d(mid_channel, out_channel, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv(x), inplace=True)
        x = F.relu(self.conv_trans(x), inplace=True)
        return x
    
class Unet_resnet18(nn.Module):
    def __init__(self, n_classes):
        super(Unet_resnet18, self).__init__()
        
        #encoder
        self.encoder = models.resnet18(pretrained=True)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1,
                                  self.encoder.relu, self.pool) #64
        self.conv2 = self.encoder.layer1 #64
        self.conv3 = self.encoder.layer2 #128
        self.conv4 = self.encoder.layer3 #256
        self.conv5 = self.encoder.layer4 #depth 512
        
        #center
        self.center = Decoder(512, 312, 256)
        
        #decoder
        self.decoder5 = Decoder(256+512, 256, 256)
        self.decoder4 = Decoder(256+256, 128, 128)
        self.decoder3 = Decoder(128+128, 64, 64)
        self.decoder2 = Decoder(64+64, 32, 32)
        self.decoder1 = Decoder(32, 16, 16)
        self.decoder0 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
    
        self.final = nn.Conv2d(8, n_classes, kernel_size=1)
        
    def forward(self, x):
        
        #encoder
        conv1 = self.conv1(x) #64x64
        conv2 = self.conv2(conv1) #32x32
        conv3 = self.conv3(conv2) #16x16
        conv4 = self.conv4(conv3) #8x8
        conv5 = self.conv5(conv4) #4x4
        
        center = self.center(self.pool(conv5)) #4x4
        #decoder
        dec5 = self.decoder5(torch.cat([center, conv5], 1)) #8x8
        dec4 = self.decoder4(torch.cat([dec5, conv4], 1)) #16x16
        dec3 = self.decoder3(torch.cat([dec4, conv3], 1)) #32x32
        dec2 = self.decoder2(torch.cat([dec3, conv2], 1)) #64x64
        dec1 = self.decoder1(dec2) #128x128
        dec0 = F.relu(self.decoder0(dec1))
        
        final = torch.sigmoid(self.final(dec0))
        
        return final
    
if __name__ == '__main__':

    
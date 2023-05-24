
# credits: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images

import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np
from torchvision.models import resnet

class vgg16_base(nn.Module):
    def __init__(self):
        super(vgg16_base,self).__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        features[0] = nn.Conv2d(13, 64, kernel_size=3, stride=1, padding=1)
        self.features = nn.ModuleList(features).eval()

    def forward(self,x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,15,22,29}:
                results.append(x)
        return results

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def conv2d_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(p=0.6),
    )

class DSIFN(nn.Module):
    def __init__(self, model_A, model_B):
        super().__init__()
        self.t1_base = model_A
        self.t2_base = model_B
        self.sa1 = SpatialAttention()
        self.sa2= SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()
        
        self.sigmoid = nn.Sigmoid()

        # channels 512, 512, 256, 128, 64
        c5 = 512
        c4 = 512
        c3 = 256
        c2 = 128
        c1 = 64
        c0 = 32

        # branch1
        self.ca1 = ChannelAttention(in_channels=c5*2)
        self.bn_ca1 = nn.BatchNorm2d(c5*2)
        self.o1_conv1 = conv2d_bn(c5*2, c5)
        self.o1_conv2 = conv2d_bn(c5, c4)
        self.bn_sa1 = nn.BatchNorm2d(c4)
        self.o1_conv3 = nn.Conv2d(c4, 1, 1)
        self.trans_conv1 = nn.ConvTranspose2d(c4, c4, kernel_size=2, stride=2)

        # branch 2
        self.ca2 = ChannelAttention(in_channels=c4*3)
        self.bn_ca2 = nn.BatchNorm2d(c4*3)
        self.o2_conv1 = conv2d_bn(c4*3, c4)
        self.o2_conv2 = conv2d_bn(c4, c3)
        self.o2_conv3 = conv2d_bn(c3, c3)
        self.bn_sa2 = nn.BatchNorm2d(c3)
        self.o2_conv4 = nn.Conv2d(c3, 1, 1)
        self.trans_conv2 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2)

        # branch 3
        self.ca3 = ChannelAttention(in_channels=c3*3)
        self.o3_conv1 = conv2d_bn(c3*3, c3)
        self.o3_conv2 = conv2d_bn(c3, c2)
        self.o3_conv3 = conv2d_bn(c2, c2)
        self.bn_sa3 = nn.BatchNorm2d(c2)
        self.o3_conv4 = nn.Conv2d(c2, 1, 1)
        self.trans_conv3 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)

        # branch 4
        self.ca4 = ChannelAttention(in_channels=c2*3)
        self.o4_conv1 = conv2d_bn(c2*3, c2)
        self.o4_conv2 = conv2d_bn(c2, c1)
        self.o4_conv3 = conv2d_bn(c1, c1)
        self.bn_sa4 = nn.BatchNorm2d(c1)
        self.o4_conv4 = nn.Conv2d(c1, 1, 1)
        self.trans_conv4 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)

        # branch 5
        self.ca5 = ChannelAttention(in_channels=c1*3)
        self.o5_conv1 = conv2d_bn(c1*3, c1)
        self.o5_conv2 = conv2d_bn(c1, c0)
        self.o5_conv3 = conv2d_bn(c0, c0)
        self.bn_sa5 = nn.BatchNorm2d(c0)
        self.o5_conv4 = nn.Conv2d(c0, 1, 1)

    def forward(self,t1_input,t2_input):
        t1_list = self.t1_base(t1_input)
        t2_list = self.t2_base(t2_input)

        t1_f_l3,t1_f_l8,t1_f_l15,t1_f_l22,t1_f_l29 = t1_list[0],t1_list[1],t1_list[2],t1_list[3],t1_list[4]
        t2_f_l3,t2_f_l8,t2_f_l15,t2_f_l22,t2_f_l29,= t2_list[0],t2_list[1],t2_list[2],t2_list[3],t2_list[4]

        x = torch.cat((t1_f_l29,t2_f_l29),dim=1)
        #optional to use channel attention module in the first combined feature
        # x = self.ca1(x) * x
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)

        branch_1_out = self.sigmoid(self.o1_conv3(x))

        x = self.trans_conv1(x)
        x = torch.cat((x,t1_f_l22,t2_f_l22),dim=1)
        x = self.ca2(x)*x
        #According to the amount of the training data, appropriately reduce the use of conv layers to prevent overfitting
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) *x
        x = self.bn_sa2(x)

        branch_2_out = self.sigmoid(self.o2_conv4(x))

        x = self.trans_conv2(x)
        x = torch.cat((x,t1_f_l15,t2_f_l15),dim=1)
        x = self.ca3(x)*x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) *x
        x = self.bn_sa3(x)

        branch_3_out = self.sigmoid(self.o3_conv4(x))

        x = self.trans_conv3(x)
        x = torch.cat((x,t1_f_l8,t2_f_l8),dim=1)
        x = self.ca4(x)*x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) *x
        x = self.bn_sa4(x)

        branch_4_out = self.sigmoid(self.o4_conv4(x))

        x = self.trans_conv4(x)
        x = torch.cat((x,t1_f_l3,t2_f_l3),dim=1)
        x = self.ca5(x)*x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) *x
        x = self.bn_sa5(x)

        branch_5_out = self.sigmoid(self.o5_conv4(x))

        return branch_5_out,branch_4_out,branch_3_out,branch_2_out,branch_1_out
    
if __name__ == '__main__':
    
    x1 = torch.rand(1,13,256,256)
    x2 = torch.rand(1,13,256,256)
    model1 = vgg16_base()
    model2 = vgg16_base()
    
    model = DSIFN(model_A=model1,
                  model_B=model2)
    out = model(x1, x2)
    
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    print(out[4].shape)
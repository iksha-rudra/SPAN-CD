
# credits: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images

from models.switchable_norm import SwitchNorm2d

import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np
from torchvision.models import resnet
from SSL.model.barlow_twins import BarlowTwins
from models.CBAM import CBAM


import os

PATH = '/home/rakesh/LocalRepo/checkpoints_300/checkpoint.pth'

class resnet18_base(nn.Module):
    def __init__(self, in_c, backbone_type = 'random'):
        super(resnet18_base,self).__init__()
        self.in_c = in_c
        
        self.feature_indices = (0, 4, 5, 6, 7)
        
        if backbone_type == 'random':
            self.encoder = resnet.resnet18(weights=None)
        elif backbone_type == 'imagenet':
            self.encoder = resnet.resnet18(weights='IMAGENET1K_V1')
        elif backbone_type == 'pretrained':
            self.encoder = resnet.resnet18(weights=None)
                    
        self.encoder.conv1 = torch.nn.Conv2d(self.in_c, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        if backbone_type == 'pretrained':
            # load from pre-trained
            # model = BarlowTwins()
            
            # if os.path.isfile(PATH):
            #     print("=> loading checkpoint '{}'".format(PATH))
            #     ckpt = torch.load(PATH, 
            #                     map_location='cpu')
            #     msg = model.load_state_dict(ckpt['model'])
            #     print(msg.missing_keys)
            #     print("=> loaded pre-trained model '{}'".format(PATH))

            #     msg = self.encoder.load_state_dict(model.backbone.state_dict(), strict=False)
            #     print(msg.missing_keys)

            # else:
            #     print("=> no checkpoint found at '{}'".format(PATH))
        
            if os.path.isfile(PATH):
                print("=> loading checkpoint '{}'".format(PATH))
                checkpoint = torch.load(PATH, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint

                msg = self.encoder.load_state_dict(state_dict, strict=False)

                print(msg.missing_keys)

                print("=> loaded pre-trained model '{}'".format(PATH))
            else:
                print("=> no checkpoint found at '{}'".format(PATH))
        
    def forward(self,x):
        feats = [x]
        for i, module in enumerate(self.encoder.children()):
            x = module(x)
            if i in self.feature_indices:
                feats.append(x)

            if i == self.feature_indices[-1]:
                break

        feats = feats[1:]
        return feats

class vgg16_base(nn.Module):
    def __init__(self):
        super(vgg16_base,self).__init__()
        features = list(vgg16(pretrained=True).features)[:30]
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
        self.relu1 = nn.GELU()
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
        SwitchNorm2d(out_channels),
        nn.Dropout(p=0.6),
    )

class DSIFN(nn.Module):
    def __init__(self, model_A, out_c):
        super().__init__()
        self.t1_base = model_A
        # self.t2_base = model_B
        self.out_c = out_c
        self.sa1 = SpatialAttention()
        self.sa2= SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()
        
        self.sigmoid = nn.Sigmoid()

        # channels 512, 512, 256, 128, 64
        c5 = 512
        c4 = 256
        c3 = 128
        c2 = 64
        c1 = 64
        c0 = 32
        c00 = 16
        
        self.cbam1 = CBAM(c5)
        self.cbam2 = CBAM(c4)
        self.cbam3 = CBAM(c3)
        self.cbam4 = CBAM(c2)
        self.cbam5 = CBAM(c1)

        # branch1
        self.ca1 = ChannelAttention(in_channels=c5*2)
        self.bn_ca1 = SwitchNorm2d(c5*2)
        self.o1_conv1 = conv2d_bn(c5*2, c5)
        self.o1_conv2 = conv2d_bn(c5, c4)
        self.bn_sa1 = SwitchNorm2d(c4)
        self.o1_conv3 = nn.Conv2d(c4, self.out_c, 1)
        self.trans_conv1 = nn.ConvTranspose2d(c4, c4, kernel_size=2, stride=2)

        # branch 2
        self.ca2 = ChannelAttention(in_channels=c4*3)
        self.bn_ca2 = SwitchNorm2d(c4*3)
        self.o2_conv1 = conv2d_bn(c4*3, c4)
        self.o2_conv2 = conv2d_bn(c4, c3)
        self.o2_conv3 = conv2d_bn(c3, c3)
        self.bn_sa2 = SwitchNorm2d(c3)
        self.o2_conv4 = nn.Conv2d(c3, self.out_c, 1)
        self.trans_conv2 = nn.ConvTranspose2d(c3, c3, kernel_size=2, stride=2)

        # branch 3
        self.ca3 = ChannelAttention(in_channels=c3*3)
        self.o3_conv1 = conv2d_bn(c3*3, c3)
        self.o3_conv2 = conv2d_bn(c3, c2)
        self.o3_conv3 = conv2d_bn(c2, c2)
        self.bn_sa3 = SwitchNorm2d(c2)
        self.o3_conv4 = nn.Conv2d(c2, self.out_c, 1)
        self.trans_conv3 = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)

        # branch 4
        self.ca4 = ChannelAttention(in_channels=c2*3)
        self.o4_conv1 = conv2d_bn(c2*3, c2)
        self.o4_conv2 = conv2d_bn(c2, c1)
        self.o4_conv3 = conv2d_bn(c1, c1)
        self.bn_sa4 = SwitchNorm2d(c1)
        self.o4_conv4 = nn.Conv2d(c1, self.out_c, 1)
        self.trans_conv4 = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)

        # branch 5
        self.ca5 = ChannelAttention(in_channels=c1*3)
        self.o5_conv1 = conv2d_bn(c1*3, c1)
        self.o5_conv2 = conv2d_bn(c1, c0)
        self.o5_conv3 = conv2d_bn(c0, c0)
        self.bn_sa5 = SwitchNorm2d(c0)
        self.o5_conv4 = nn.Conv2d(c0, self.out_c, 1)         #Last conv
        
        ''' Extra '''
        self.trans_conv5 = nn.ConvTranspose2d(c0, c0, kernel_size=2, stride=2)
        # branch 6
        self.o6_conv2 = conv2d_bn(c0, c00)
        self.o6_conv3 = conv2d_bn(c00, c00)
        self.bn_sa6 = SwitchNorm2d(c00)
        self.o6_conv4 = nn.Conv2d(c00, self.out_c, 1)
        
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self,t1_input,t2_input):
        t1_list = self.t1_base(t1_input)
        t2_list = self.t1_base(t2_input)

        t1_f_l3,t1_f_l8,t1_f_l15,t1_f_l22,t1_f_l29 = t1_list[0],t1_list[1],t1_list[2],t1_list[3],t1_list[4]
        t2_f_l3,t2_f_l8,t2_f_l15,t2_f_l22,t2_f_l29 = t2_list[0],t2_list[1],t2_list[2],t2_list[3],t2_list[4]

        x = torch.cat((t1_f_l29,t2_f_l29),dim=1)
        #optional to use channel attention module in the first combined feature
        x = self.ca1(x) * x
        x = self.o1_conv1(x)
        # x = self.cbam1(x)
        x = self.o1_conv2(x)
        x = self.sa1(x) * x
        x = self.bn_sa1(x)

        branch_1_out = self.sm(self.o1_conv3(x))
        # branch_1_out = self.o1_conv3(x)

        x = self.trans_conv1(x)
        x = torch.cat((x,t1_f_l22,t2_f_l22),dim=1)
        x = self.ca2(x)*x
        #According to the amount of the training data, appropriately reduce the use of conv layers to prevent overfitting
        x = self.o2_conv1(x)
        # x = self.cbam2(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa2(x) *x
        x = self.bn_sa2(x)

        branch_2_out = self.sm(self.o2_conv4(x))
        # branch_2_out = self.o2_conv4(x)

        x = self.trans_conv2(x)
        x = torch.cat((x,t1_f_l15,t2_f_l15),dim=1)
        x = self.ca3(x)*x
        x = self.o3_conv1(x)
        # x = self.cbam3(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa3(x) *x
        x = self.bn_sa3(x)

        branch_3_out = self.sm(self.o3_conv4(x))
        # branch_3_out = self.o3_conv4(x)

        x = self.trans_conv3(x)
        x = torch.cat((x,t1_f_l8,t2_f_l8),dim=1)
        x = self.ca4(x)*x
        x = self.o4_conv1(x)
        # x = self.cbam4(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa4(x) *x
        x = self.bn_sa4(x)

        branch_4_out = self.sm(self.o4_conv4(x))
        # branch_4_out = self.o4_conv4(x)

        x = self.trans_conv4(x)
        x = torch.cat((x,t1_f_l3,t2_f_l3),dim=1)
        x = self.ca5(x)*x
        x = self.o5_conv1(x)
        # x = self.cbam5(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa5(x) *x
        x = self.bn_sa5(x)

        branch_5_out = self.sm(self.o5_conv4(x))
        # branch_5_out = self.o5_conv4(x)
        
        ''' Extra '''
        x = self.trans_conv5(x)

        x = self.o6_conv2(x)
        x = self.o6_conv3(x)
        x = self.bn_sa6(x)
        branch_6_out = self.sm(self.o6_conv4(x))
        # branch_6_out = self.o6_conv4(x)
        
        return [branch_6_out, branch_5_out,branch_4_out,branch_3_out,branch_2_out,branch_1_out]
        
        # return branch_6_out

class DSIFN_ResNet(nn.Module):
    """Some Information about DSIFN_ResNet"""
    def __init__(self, in_c, out_c,  backbone_type = 'random', finetune = True):
        super(DSIFN_ResNet, self).__init__()
        
        model1 = resnet18_base(in_c, backbone_type)
        # model2 = resnet18_base(in_c, backbone_type)
        
        for param in model1.parameters():
            param.requires_grad = finetune 

        self.model = DSIFN(model_A=model1,
                        out_c=out_c)

    def forward(self, x1, x2):

        out = self.model(x1, x2)
        return out
    
if __name__ == '__main__':
    
    x1 = torch.rand(1,13,256,256)
    x2 = torch.rand(1,13,256,256)
    # model1 = vgg16_base()
    # model2 = vgg16_base()
    
    # model1 = resnet18_base()
    # model2 = resnet18_base()
    
    # model = DSIFN(model_A=model1,
    #               model_B=model2)
    # out = model(x1, x2)
    
    # print(out.shape)
    # print(out[1].shape)
    # print(out[2].shape)
    # print(out[3].shape)
    # print(out[4].shape)
    # print(out[5].shape)
    
    model = DSIFN_ResNet(in_c=13, out_c=2, backbone_type = 'pretrained', finetune = False)
    out = model(x1, x2)
    
    print(out.shape)

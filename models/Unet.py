import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.padding import ReplicationPad2d
from models.switchable_norm import SwitchNorm2d
from models.CBAM import CBAM

class Convolution2D(nn.Module):
    """Some Information about ConvRelu"""
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(Convolution2D, self).__init__()
        
        self.Convolve2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.Convolve2d(x)
        output = self.relu(out)
        
        return output
    
class TransConvolution2D(nn.Module):
    """Some Information about ConvRelu"""
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(TransConvolution2D, self).__init__()
        
        self.Convolve2d = nn.ConvTranspose2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.Convolve2d(x)
        output = self.relu(out)
        
        return output
    
class Pooling_Block(nn.Module):
    """Some Information about Pooling_Block"""
    def __init__(self, in_channel, out_channel, stride):
        super(Pooling_Block, self).__init__()
        
        self.strided_Conv = Convolution2D(in_channels=in_channel,
                                            out_channels=out_channel,
                                            kernel_size=3,
                                            padding=1,
                                            stride=stride)
        self.AvgPool = nn.AvgPool2d(kernel_size=stride)
        self.pointConv = Convolution2D(in_channels=in_channel,
                                       out_channels=out_channel,
                                       kernel_size=1,
                                       padding=0,
                                       stride=1)

    def forward(self, x):
        out_1 = self.strided_Conv(x)

        out_2 = self.AvgPool(x)
        out_2 = self.pointConv(out_2)
        
        output = out_1 + out_2
        return output
    
class Gobal_Pooling(nn.Module):
    """Some Information about Gobal_Pooling"""
    def __init__(self, in_channel, size):
        super(Gobal_Pooling, self).__init__()
        
        self.ReduceMean = nn.AdaptiveAvgPool2d(2)
        self.pointConv = Convolution2D(in_channels=in_channel,
                                   out_channels=in_channel,
                                   kernel_size=1,
                                   padding=0,
                                   stride=1
                                   )
        
        self.upSample =TransConvolution2D(in_channels=in_channel,
                                           out_channels=in_channel,
                                           kernel_size=3,
                                           padding=1,
                                           stride=size-1
                                           )
        
    def forward(self, x):
        out = self.ReduceMean(x)
        out = self.pointConv(out)
        out = self.upSample(out)
        
        return out

class NSPP_Block(nn.Module):
    """Some Information about NSPP_Block"""
    def __init__(self, channel, size):
        super(NSPP_Block, self).__init__()

        self.pool2 = Pooling_Block(channel, channel//4, 2)
        self.pool4 = Pooling_Block(channel, channel//4, 4)
        self.pool8 = Pooling_Block(channel, channel//4, 8)
        self.pool16 = Pooling_Block(channel, channel//4, 16)
        
        self.global_pool = Gobal_Pooling(channel//4, size)

    def forward(self, x):
        
        p2 = self.pool2(x)
        p4 = self.pool4(x)
        p8 = self.pool8(x)
        p16 = self.pool16(x)
        
        x1 = self.global_pool(p2)
        x2 = self.global_pool(p4)
        x3 = self.global_pool(p8)
        x4 = self.global_pool(p16)
        
        feats = torch.cat([x1, x2, x3, x4],axis=1)
    
        return x
    
class Unet(nn.Module):
    """EF segmentation network."""

    def __init__(self, input_nbr, label_nbr):
        super(Unet, self).__init__()

        self.use_dropout = True
        self.use_batch_norm = True
        
        self.input_nbr = input_nbr

        self.conv11 = nn.Conv2d(input_nbr, 16, kernel_size=3, padding=1)
        self.bn11 = self.make_batch_norm(16)
        self.do11 = self.make_dropout()   
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = self.make_batch_norm(16)
        self.do12 = self.make_dropout()   

        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = self.make_batch_norm(32)
        self.do21 = self.make_dropout()   
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = self.make_batch_norm(32)
        self.do22 = self.make_dropout()   

        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = self.make_batch_norm(64)
        self.do31 = self.make_dropout()   
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = self.make_batch_norm(64)
        self.do32 = self.make_dropout()   
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = self.make_batch_norm(64)
        self.do33 = self.make_dropout()   

        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = self.make_batch_norm(128)
        self.do41 = self.make_dropout()   
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = self.make_batch_norm(128)
        self.do42 = self.make_dropout()   
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = self.make_batch_norm(128)
        self.do43 = self.make_dropout()   


        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn43d = self.make_batch_norm(128)
        self.do43d = self.make_dropout()   
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = self.make_batch_norm(128)
        self.do42d = self.make_dropout()   
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = self.make_batch_norm(64)
        self.do41d = self.make_dropout()   

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn33d = self.make_batch_norm(64)
        self.do33d = self.make_dropout()   
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = self.make_batch_norm(64)
        self.do32d = self.make_dropout()   
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = self.make_batch_norm(32)
        self.do31d = self.make_dropout()   

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn22d = self.make_batch_norm(32)
        self.do22d = self.make_dropout()   
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = self.make_batch_norm(16)
        self.do21d = self.make_dropout()   

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn12d = self.make_batch_norm(16)
        self.do12d = self.make_dropout()   
        self.conv11d = nn.ConvTranspose2d(16, label_nbr, kernel_size=3, padding=1)

        self.sm = nn.LogSoftmax(dim=1)

        self.nspp = NSPP_Block(128, 128)

        self.cbam16 = CBAM(16)
        self.cbam32 = CBAM(32)
        self.cbam64 = CBAM(64)
        self.cbam128 = CBAM(128)

        # self.initialize_weights()

    def forward(self, x1, x2):

        x = torch.cat((x1, x2), 1)

        """Forward method."""
        # Stage 1
        x11 = self.do11(F.gelu(self.bn11(self.conv11(x))))
        x12 = self.do12(F.gelu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.gelu(self.bn21(self.conv21(x1p))))
        x22 = self.do22(F.gelu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.gelu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.gelu(self.bn32(self.conv32(x31))))
        x33 = self.do33(F.gelu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.gelu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.gelu(self.bn42(self.conv42(x41))))
        x43 = self.do43(F.gelu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43, kernel_size=2, stride=2)


        x4p_nspp = self.nspp(x4p)

        # Stage 4d
        x4d = self.upconv4(x4p_nspp)
        pad4 = ReplicationPad2d((0, x43.size(3) - x4d.size(3), 0, x43.size(2) - x4d.size(2)))

        x43_cbam = self.cbam128(x43)

        x4d = torch.cat((pad4(x4d), x43_cbam), 1)
        x43d = self.do43d(F.gelu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.gelu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.gelu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33.size(3) - x3d.size(3), 0, x33.size(2) - x3d.size(2)))

        x33_cbam = self.cbam64(x43)

        x3d = torch.cat((pad3(x3d), x33_cbam), 1)
        x33d = self.do33d(F.gelu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.gelu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.gelu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22.size(3) - x2d.size(3), 0, x22.size(2) - x2d.size(2)))

        x22_cbam = self.cbam32(x22)

        x2d = torch.cat((pad2(x2d), x22_cbam), 1)
        x22d = self.do22d(F.gelu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.gelu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12.size(3) - x1d.size(3), 0, x12.size(2) - x1d.size(2)))

        x12_cbam = self.cbam16(x12)

        x1d = torch.cat((pad1(x1d), x12_cbam), 1)
        x12d = self.do12d(F.gelu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        # return self.sm(x11d)
        return x11d
    
    def make_dropout(self):
        if self.use_dropout:
            return nn.Dropout2d(p=0.2)
        else:
            return nn.Identity()
        
    def make_batch_norm(self, channels):
        if self.use_batch_norm:
            return nn.BatchNorm2d(channels)
            # return SwitchNorm2d(channels)
        else:
            return nn.Identity()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                
                # if m.bias is not None:
                #     nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias,0)
                
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                # nn.init.constant_(m.bias, 0)
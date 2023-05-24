import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class UCDNet(nn.Module):
    """ Urban Change Detection Net """
    def __init__(self, input_nbr, output_nbr):
        super(UCDNet, self).__init__()
        
        ''' Encoder Stage 1 '''
        self.conv_11e = Convolution2D(input_nbr, 16, kernel_size=3, padding=1)
        self.conv_12e = Convolution2D(16, 16, kernel_size=3, padding=1)
        self.conv_1feat = Convolution2D(16, 16, kernel_size=1, padding=0)
        self.max_pool_1e = nn.MaxPool2d(kernel_size=2)
        
        ''' Encoder State 2 '''
        self.conv_21e = Convolution2D(32, 32, kernel_size=3, padding=1)
        self.conv_22e = Convolution2D(32, 32, kernel_size=3, padding=1)
        self.conv_2feat = Convolution2D(32, 32, kernel_size=1, padding=0)
        self.max_pool_2e = nn.MaxPool2d(kernel_size=2)
        
        ''' Encoder State 3 '''
        self.conv_31e = Convolution2D(64, 64, kernel_size=3, padding=1)
        self.conv_32e = Convolution2D(64, 64, kernel_size=3, padding=1)
        self.conv_33e = Convolution2D(64, 64, kernel_size=3, padding=1)
        self.conv_3feat = Convolution2D(64, 64, kernel_size=1, padding=0)
        self.max_pool_3e = nn.MaxPool2d(kernel_size=2)
        
        ''' Encoder State 4 ''' 
        self.conv_41e = Convolution2D(128, 128, kernel_size=3, padding=1)
        self.conv_42e = Convolution2D(128, 128, kernel_size=3, padding=1)
        self.conv_43e = Convolution2D(128, 128, kernel_size=3, padding=1)
        self.conv_4feat = Convolution2D(128, 128, kernel_size=1, padding=0)
        self.conv_45e = Convolution2D(256, 64, kernel_size=3, padding=1)
        
        ''' Decoder State 1 ''' 
        self.up_11d = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_11d = Convolution2D(192, 64, kernel_size=1, padding=0)
        self.conv_12d = Convolution2D(256, 64, kernel_size=3, padding=1)
        self.conv_13d = Convolution2D(64, 64, kernel_size=3, padding=1)
        self.conv_14d = Convolution2D(64, 32, kernel_size=3, padding=1)
        self.bn_1d = nn.BatchNorm2d(32)
        
        ''' Decoder State 2 '''
        self.up_21d = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_21d = Convolution2D(32, 32, kernel_size=1, padding=0)
        self.conv_22d = Convolution2D(128, 32, kernel_size=3, padding=1)
        self.conv_23d = Convolution2D(32, 16, kernel_size=3, padding=1)
        self.bn_2d = nn.BatchNorm2d(16) 
        
        ''' Decoder State 3 ''' 
        self.up_31d = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_31d = Convolution2D(16, 16, kernel_size=1, padding=0)
        self.conv_32d = Convolution2D(64, 16, kernel_size=3, padding=1)
        self.bn_3d = nn.BatchNorm2d(16) 
        
        self.conv_33d = Convolution2D(16, output_nbr, kernel_size=3, padding=1)
        self.sm = nn.LogSoftmax(dim=1)
        
        self.nspp = NSPP_Block(192, 64)
        
        # self.initialize_weights()
        

    def forward(self, x1, x2):

        '''  Ecoder Stage 1 '''
        e1_s1_c1 = self.conv_11e(x1)
        e1_s1_c2 = self.conv_12e(e1_s1_c1)

        e2_s1_c1 = self.conv_11e(x2)
        e2_s1_c2 = self.conv_12e(e2_s1_c1)
        
        s1_feat = torch.abs(e1_s1_c1 - e2_s1_c1)
        refined_s1_feat = self.conv_1feat(s1_feat)
        
        e1_s1_skip = torch.cat([e1_s1_c2, refined_s1_feat ], axis=1)
        e2_s1_skip = torch.cat([e2_s1_c2, refined_s1_feat ], axis=1)
        s1_skip = torch.cat([e1_s1_c2, refined_s1_feat, e2_s1_c2], axis=1)
        
        e1_s1_mp = self.max_pool_1e(e1_s1_skip)
        e2_s1_mp = self.max_pool_1e(e2_s1_skip)
        
        ''' Ecoder Stage 2 '''
        e1_s2_c1 = self.conv_21e(e1_s1_mp)
        e1_s2_c2 = self.conv_22e(e1_s2_c1)

        e2_s2_c1 = self.conv_21e(e2_s1_mp)
        e2_s2_c2 = self.conv_22e(e2_s2_c1)
        
        s2_feat = torch.abs(e1_s2_c1 - e2_s2_c1)
        refined_s2_feat = F.relu(self.conv_2feat(s2_feat))
        
        e1_s2_skip = torch.cat([e1_s2_c2, refined_s2_feat ], axis=1)
        e2_s2_skip = torch.cat([e2_s2_c2, refined_s2_feat ], axis=1)
        s2_skip = torch.cat([e1_s2_c2, refined_s2_feat, e2_s2_c2], axis=1)
        
        e1_s2_mp = self.max_pool_2e(e1_s2_skip)
        e2_s2_mp = self.max_pool_2e(e2_s2_skip)                
        
        ''' Ecoder Stage 3 '''
        e1_s3_c1 = self.conv_31e(e1_s2_mp)
        e1_s3_c2 = self.conv_32e(e1_s3_c1)
        e1_s3_c3 = self.conv_32e(e1_s3_c2)

        e2_s3_c1 = self.conv_31e(e2_s2_mp)
        e2_s3_c2 = self.conv_32e(e2_s3_c1)
        e2_s3_c3 = self.conv_32e(e2_s3_c2)
        
        s3_feat = torch.abs(e1_s3_c1 - e2_s3_c1)
        refined_s3_feat = self.conv_3feat(s3_feat)
        
        e1_s3_skip = torch.cat([e1_s3_c3, refined_s3_feat ], axis=1)
        e2_s3_skip = torch.cat([e2_s3_c3, refined_s3_feat ], axis=1)
        s3_skip = torch.cat([e1_s3_c3, refined_s3_feat, e2_s3_c3], axis=1)
        
        e1_s3_mp = self.max_pool_3e(e1_s3_skip)
        e2_s3_mp = self.max_pool_3e(e2_s3_skip)        
        
        ''' Ecoder Stage 4  '''
        e1_s4_c1 = self.conv_41e(e1_s3_mp)
        e1_s4_c2 = self.conv_42e(e1_s4_c1)
        e1_s4_c3 = self.conv_42e(e1_s4_c2)

        e2_s4_c1 = self.conv_41e(e2_s3_mp)
        e2_s4_c2 = self.conv_42e(e2_s4_c1)
        e2_s4_c3 = self.conv_42e(e2_s4_c2)
        
        s4_feat = torch.abs(e1_s4_c1 - e2_s4_c1)
        refined_s4_feat = self.conv_4feat(s4_feat)
        
        e1_s4_skip = torch.cat([e1_s4_c3, refined_s4_feat ], axis=1)
        e2_s4_skip = torch.cat([e2_s4_c3, refined_s4_feat ], axis=1)
        
        e1_s4_fmp = self.conv_45e(e1_s4_skip)
        e2_s4_fmp = self.conv_45e(e2_s4_skip)
        
        s4_feat_2 = torch.abs(e1_s4_fmp - e2_s4_fmp)
        s4_skip = torch.cat([e1_s4_fmp, s4_feat_2, e2_s4_fmp],axis=1)
        
        nspp_s4_skip = self.nspp(s4_skip)
        
        ''' Decoder State 1 ''' 
        d_s1_up = self.up_11d(nspp_s4_skip)
        d_s1_c1 = self.conv_11d(d_s1_up)
        
        cc_feat = torch.cat([d_s1_c1, s3_skip], axis=1)
        d_s1_c2 = self.conv_12d(cc_feat)
        d_s1_c3 = self.conv_13d(d_s1_c2)
        d_s1_c4 = self.conv_14d(d_s1_c3)
        d_s1_bn = self.bn_1d(d_s1_c4)
        
        ''' Decoder State 2 '''
        d_s2_up = self.up_21d(d_s1_bn)
        d_s2_c1 = self.conv_21d(d_s2_up)
        cc_feat = torch.cat([d_s2_c1, s2_skip], axis=1)
        
        d_s2_c2 = self.conv_22d(cc_feat)
        d_s2_c3 = self.conv_23d(d_s2_c2)
        d_s2_bn = self.bn_2d(d_s2_c3)
        
        ''' Decoder State 3 ''' 
        d_s3_up = self.up_31d(d_s2_bn)
        d_s3_c1 = self.conv_31d(d_s3_up)
        cc_feat = torch.cat([d_s3_c1, s1_skip], axis=1)
        d_s3_c2 = self.conv_32d(cc_feat)
        d_s3_bn = self.bn_3d(d_s3_c2)
        
        d_s3_c3 = self.conv_33d(d_s3_bn)
        # output = self.sm(d_s3_c3)
        
        # return output
        return d_s3_c3
    
    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_uniform_(m.weight)
                
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias,0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias,0)
                
    #         elif isinstance(m, nn.Linear):
    #             nn.init.kaiming_uniform_(m.weight)
    #             nn.init.constant_(m.bias, 0)
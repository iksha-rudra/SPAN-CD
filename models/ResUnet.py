import torch
import torch.nn as nn
from models.switchable_norm import SwitchNorm2d

class batchnorm_relu(nn.Module):
    def __init__(self, in_c) -> None:
        super().__init__()
        # self.bn = nn.BatchNorm2d(in_c)
        self.bn = SwitchNorm2d(in_c)
        # self.relu = nn.ReLU()
        self.relu = nn.GELU()
    
    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        
        ''' Convolutional Layer '''
        
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)
        
        ''' Shortcut connection '''
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)
        
    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)
        
        skip = x + s
        return skip
        
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.r = residual_block(in_c + out_c, out_c)
        
    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        
        return x

class ResUNet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        ''' Encoder 1 '''
        self.c11 = nn.Conv2d(in_c, 16, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(16)
        self.c12 = nn.Conv2d(16,16, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(in_c, 16, kernel_size=1, padding=0)
    
        ''' Encoder 2 and 3 '''
        self.r2 = residual_block(16, 32, stride=2)
        self.r3 = residual_block(32, 64, stride=2)
        
        ''' Bottle Neck '''
        self.r4 = residual_block(64,128, stride=2)
        
        ''' Decoder '''
        self.d1 = decoder_block(128, 64)
        self.d2 = decoder_block(64,32)
        self.d3 = decoder_block(32,16)
        
        ''' Output '''
        self.output = nn.Conv2d(16, out_c, kernel_size=1, padding=0)
        self.sm = nn.LogSoftmax(dim=1)
        
        # self.initialize_weights()
                
    
    def forward(self, inputs):
           
        ''' Encoder 1 '''
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s
        
        ''' Encoder 2 and 3 '''
        skip2 = self.r2(skip1) 
        skip3 = self.r3(skip2)
        
        ''' Bottle Neck '''
        b = self.r4(skip3)
        
        ''' Decoder '''
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)
        
        ''' Output '''
        output = self.output(d3)
        
        return output
    
# if __name__ == '__main__':
#     model = ResUNet(26, 2)
#     img1 = torch.rand(1,13,256,256)
#     img2 = torch.rand(1,13,256,256)
#     img_combined = torch.cat([img1, img2], dim=1)
#     out = model(img_combined)
#     print("==>> out: ", out.shape)
    
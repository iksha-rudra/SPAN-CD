import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    
    def forward(self, x):
        return self.bn(self.c(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        res_channels = out_channels
        stride = 1

        self.projection = in_channels!=out_channels
        
        if self.projection:
            self.p = ConvBlock(in_channels, out_channels, 1, 2, 0)
            stride = 2

        self.c1 = ConvBlock(in_channels, res_channels, 3, stride, 1)
        self.c2 = ConvBlock(res_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        f = self.relu(self.c1(x))
        f = self.c2(f)

        if self.projection:
            x = self.p(x)

        h = self.relu(torch.add(f, x))
        return h
    
class ResCDNet34(nn.Module):
    def __init__(
        self, 
        in_channels=13,
        out_channels=2
        ):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 16, 3, 1, 1)
        self.conv1 = ConvBlock(32, 32, 3, 1, 1)
        
        self.conv1 = ConvBlock(in_channels, 32, 3, 1, 1)
        
        ResidualBlock(64, 64)
        ResidualBlock(64, 64)
        
        self.resblock_11 = ResidualBlock(32, 64)
        # self.resblock_12 = ResidualBlock(64, 64)
        self.resblock_13 = ResidualBlock(64, 64)   
        
        self.resblock_21 = ResidualBlock(64, 128)
        # self.resblock_22 = ResidualBlock(128, 128) 
        # self.resblock_23 = ResidualBlock(128, 128) 
        self.resblock_24 = ResidualBlock(128, 128)
        
        self.resblock_31 = ResidualBlock(128, 256)
        # self.resblock_32 = ResidualBlock(256, 256)
        # self.resblock_33 = ResidualBlock(256, 256)
        # self.resblock_34 = ResidualBlock(256, 256)      
        # self.resblock_35 = ResidualBlock(256, 256)
        self.resblock_36 = ResidualBlock(256, 256)
        
        self.resblock_41 = ResidualBlock(256, 512)
        # self.resblock_42 = ResidualBlock(512, 512)
        self.resblock_43 = ResidualBlock(512, 512)

        self.relu = nn.ReLU()
        
        self.decoder_11 = ConvBlock(768, 256, 3, 1, 1)
        self.decoder_12 = ConvBlock(256, 256, 3, 1, 1)

        self.decoder_21 = ConvBlock(384, 128, 3, 1, 1)
        self.decoder_22 = ConvBlock(128, 128, 3, 1, 1)

        self.decoder_31 = ConvBlock(192, 64, 3, 1, 1)
        self.decoder_32 = ConvBlock(64, 64, 3, 1, 1)

        self.decoder_41 = ConvBlock(128, 32, 3, 1, 1)
        self.decoder_42 = ConvBlock(32, 32, 3, 1, 1)
        
        self.decoder_51 = ConvBlock(32, 16, 3, 1, 1)
        self.decoder_52 = ConvBlock(16, 16, 3, 1, 1)
        self.decoder_53 = ConvBlock(16, out_channels, 3, 1, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # self.init_weight()

    def forward(self, x1, x2):
        
        ''' Enc 1'''
        x_10 = self.relu(self.conv1(x1))
        
        x = self.maxpool(x_10)
        
        x = self.resblock_11(x)
        # x = self.resblock_12(x)
        x_11 = self.resblock_13(x)

        x = self.resblock_21(x_11)
        # x = self.resblock_22(x)
        # x = self.resblock_23(x)
        x_12 = self.resblock_24(x)
        
        x = self.resblock_31(x_12)
        # x = self.resblock_32(x)
        # x = self.resblock_33(x)
        # x = self.resblock_34(x)
        # x = self.resblock_35(x)
        x_13 = self.resblock_36(x)

        x = self.resblock_41(x_13)
        # x = self.resblock_42(x)
        x = self.resblock_43(x)

        x_scaled = self.upsample(x)
        
        ''' Enc2 '''
        
        x_20 = self.relu(self.conv1(x2))
        
        x = self.maxpool(x_20)
        
        x = self.resblock_11(x)
        # x = self.resblock_12(x)
        x_21 = self.resblock_13(x)

        x = self.resblock_21(x_21)
        # x = self.resblock_22(x)
        # x = self.resblock_23(x)
        x_22 = self.resblock_24(x)
        
        x = self.resblock_31(x_22)
        # x = self.resblock_32(x)
        # x = self.resblock_33(x)
        # x = self.resblock_34(x)
        # x = self.resblock_35(x)
        x_23 = self.resblock_36(x)

        x = self.resblock_41(x_23)
        # x = self.resblock_42(x)
        x = self.resblock_43(x)

        x_scaled = self.upsample(x)
        
        x = self.relu(self.decoder_11(torch.cat([torch.abs(x_13 - x_23), x_scaled],dim=1)))
        x = self.relu(self.decoder_12(x))
        
        x_scaled = self.upsample(x)
         
        x = self.relu(self.decoder_21(torch.cat([torch.abs(x_12 - x_22), x_scaled],dim=1)))
        x = self.relu(self.decoder_22(x))
        
        x_scaled = self.upsample(x)
        
        x = self.relu(self.decoder_31(torch.cat([torch.abs(x_11 - x_21), x_scaled],dim=1)))
        x = self.relu(self.decoder_32(x))
        
        x_scaled = self.upsample(x)      
        
        x = self.relu(self.decoder_41(torch.cat([torch.abs(x_10 - x_20), x_scaled],dim=1)))
        x = self.relu(self.decoder_42(x))
        
        x_scaled = self.upsample(x)    
        
        x = self.relu(self.decoder_51(x_scaled))
        x = self.relu(self.decoder_52(x))  
        x = self.relu(self.decoder_53(x))                
        
        return x

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)


# if __name__ == "__main__":

#     resunet34 = ResCDNet34(13)
#     image = torch.rand(1, 13, 256, 256)
#     print(resunet34(image, image ).shape)
    
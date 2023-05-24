import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
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
    
class ResNet34(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        classes=1000
        ):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.resblock_11 = ResidualBlock(64, 64)
        self.resblock_12 = ResidualBlock(64, 64)
        self.resblock_13 = ResidualBlock(64, 64)   
        
        self.resblock_21 = ResidualBlock(64, 128)
        self.resblock_22 = ResidualBlock(128, 128) 
        self.resblock_23 = ResidualBlock(128, 128) 
        self.resblock_24 = ResidualBlock(128, 128)
        
        self.resblock_31 = ResidualBlock(128, 256)
        self.resblock_32 = ResidualBlock(256, 256)
        self.resblock_33 = ResidualBlock(256, 256)
        self.resblock_34 = ResidualBlock(256, 256)      
        self.resblock_35 = ResidualBlock(256, 256)
        self.resblock_36 = ResidualBlock(256, 256)
        
        self.resblock_41 = ResidualBlock(256, 512)
        self.resblock_42 = ResidualBlock(512, 512)
        self.resblock_43 = ResidualBlock(512, 512)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, classes)

        self.relu = nn.ReLU()

        self.init_weight()

    def forward(self, x):
        print(x.shape)
        
        x = self.relu(self.conv1(x))
        print(x.shape)
        
        x = self.maxpool(x)
        
        print(x.shape)
        
        x = self.resblock_11(x)
        x = self.resblock_12(x)
        x = self.resblock_13(x)
        
        print(x.shape)

        x = self.resblock_21(x)
        x = self.resblock_22(x)
        x = self.resblock_23(x)
        x = self.resblock_24(x)
        
        print(x.shape)
        
        x = self.resblock_31(x)
        x = self.resblock_32(x)
        x = self.resblock_33(x)
        x = self.resblock_34(x)
        x = self.resblock_35(x)
        x = self.resblock_36(x)
        
        print(x.shape)

        x = self.resblock_41(x)
        x = self.resblock_42(x)
        x = self.resblock_43(x)
        
        print(x.shape)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)


if __name__ == "__main__":

    resnet34 = ResNet34(1)
    image = torch.rand(1, 1, 256, 256)
    print(resnet34(image).shape)
    
    import segmentation_models_pytorch as smp

    model = smp.Unet()   
    print(model)
    
    # print(resnet34)
    
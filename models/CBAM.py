import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CAM(nn.Module):
    """Channel Attention Module (CAM) """
    def __init__(self, ch, ratio=8):
        super(CAM, self).__init__()
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch // ratio, bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // ratio, ch, bias = False)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x1 = self.avg(x)
        x2 = self.max(x)
        x1 = x1.squeeze(-1).squeeze(-1)
        x2 = x2.squeeze(-1).squeeze(-1)
        
        x1 = self.mlp(x1)
        x2 = self.mlp(x2)
        
        feat = x1 + x2
        feat = self.sigmoid(feat)
        feat = feat.unsqueeze(-1).unsqueeze(-1)
        
        refined_feat = x * feat

        return refined_feat
    
class SAM(nn.Module):
    """ Saptial Attention Module (SAM) """
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        
        self.conv = nn.Conv2d(2,1,kernel_size=kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2,_ = torch.max(x,dim=1, keepdim=True)
        
        feats = torch.cat([x1, x2], dim=1)
        feats = self.conv(feats)
        feats = self.sigmoid(feats)
        refined_feats = x * feats

        return refined_feats
    
class CBAM(nn.Module):
    def __init__(self, channel, kernel_size=7) -> None:
        super().__init__()
        
        self.cam = CAM(channel)
        self.sam = SAM(kernel_size)
        
    def forward(self, x):
        x = self.cam(x)
        x = self.sam(x)
        
        return x
    
# if __name__ == '__main__':
#     x = torch.rand(1,16,512,512)
#     model = CBAM(16)
#     output = model(x)
#     print(x.shape)
#     print(output.shape)
    
    
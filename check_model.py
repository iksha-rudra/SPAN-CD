import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sota_models.swin_ynet import SwinCD

def check_model():
    model = SwinCD()

    input1 = torch.rand(1,13,224,224)
    input2 = torch.rand(1,13,224,224)
    output = model(input1, input2)

    print(input1.shape)
    print(output.shape) 

if __name__ == '__main__':
    check_model()
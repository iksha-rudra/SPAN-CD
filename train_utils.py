import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from time import time, ctime
# from models.Unet import Unet
# from models.fc_siamunet_diff import SiamUNet_diff
# from models.fc_siamunet_conc import SiamUNet_conc
from models.UCDNet import UCDNet
from models.ResCDNet34 import ResCDNet34
from models.RDPNet import RDPNet
from models.ResUnet import ResUNet
from models.DSIFN_ResNet import DSIFN_ResNet
from models.DSIFN_EffNet_B0 import DSIFN_EfficientNet_B0
from models.DSIFN_EffNet_B7 import DSIFN_EfficientNet_B7
from models.DSIFN_EffNet_B4 import DSIFN_EfficientNet_B4

from sota_models.fc_ef import FC_EF
from sota_models.fc_siam_conc import SiamUNet_conc
from sota_models.fc_siam_diff import SiamUNet_diff

from sota_models.snunet import SNUNet
from sota_models.bit import BIT
from sota_models.lunet import LUNet
from sota_models.dtcdscn import CDNet_model
from sota_models.cdnet import CDNet
from sota_models.ChangeFormer import ChangeFormerV6
# from sota_models.swin_ynet import SwinCD

# from SSL.model.segmentation import SiamSegment
from SSL.model.segmentation_effnet import SiamSegment


from loss_functions.dice_loss import DiceLoss
from loss_functions.combine_loss import CombineLoss
from loss_functions.bce_loss import BCELoss
from loss_functions.ds_loss import DS_Loss_all

import segmentation_models_pytorch as smp

def create_result_directory(path):
    from datetime import datetime

    today = datetime.today()
    d4 = today.strftime("%b-%d-%Y")

    parent_path_to_save = os.path.join(path, d4)
    if not os.path.exists(parent_path_to_save):
        os.makedirs(parent_path_to_save)
        print("Directory '% s' created" % d4)

    from datetime import datetime
    currentDateAndTime = datetime.now()
    current_time = currentDateAndTime.strftime("%H_%M_%S")

    res_dir = os.path.join(parent_path_to_save, str(current_time))
    os.makedirs(res_dir) 
    
    return res_dir

def get_model(name, in_channel, out_channel):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = None

    if name == 'SiamUNet_conc':
        model = SiamUNet_conc(in_channel, out_channel)
        model = model.to(device = device)
        
    elif name == 'SiamUNet_diff':
        model = SiamUNet_diff(in_channel, out_channel)
        model = model.to(device = device)
    
    elif name == 'ResUNet':
        model = ResUNet(in_channel*2, out_channel)
        model = model.to(device = device)    
        
    elif name == 'FC_EF':
        model = FC_EF(in_channel*2, out_channel)
        model = model.to(device = device)

    elif name == 'UCDNet':
        model = UCDNet(in_channel, out_channel)
        model = model.to(device = device)
        
    elif name == 'ResCDNet34':
        model = ResCDNet34(in_channel, out_channel)
        model = model.to(device = device)
        
    elif name == 'RDPNet':
        model = RDPNet(in_channel, out_channel)
        model = model.to(device = device)
        
    elif name == 'SNUNet':
        model = SNUNet(in_channel, out_channel)
        model = model.to(device = device)
        
    elif name == 'BIT':
        model = BIT(in_channel, out_channel)
        model = model.to(device = device)   
        
    elif name == 'LUNet':
        model = LUNet(in_channel, out_channel)
        model = model.to(device = device)
        
    elif name == 'DTCDSDN':
        model = CDNet_model(in_channel)
        model = model.to(device = device)
        
    elif name == 'CDNet':
        model = CDNet(in_channel, out_channel)
        model = model.to(device = device)

    elif name == 'ChangeFormer':
        model = ChangeFormerV6(in_channel, out_channel, decoder_softmax=True)
        model = model.to(device = device)
              
    # elif name == 'SwinYNet':
    #     model = SwinCD()
    #     model = model.to(device = device)      

    elif name == 'SiamSegment':
        model = SiamSegment(backbone_type='random')
        model = model.to(device=device)  
        
    elif name == 'DSIFN_ResNet':
        model = DSIFN_ResNet(in_channel, out_channel, 
                             backbone_type='pretrained',
                             finetune=True)
        model = model.to(device=device)          
        
    elif name == 'DSIFN_EfficientNet_B0':
        model = DSIFN_EfficientNet_B0(in_channel, out_channel, 
                             backbone_type='random',
                             finetune=True)
        model = model.to(device=device)  
    
    elif name == 'DSIFN_EfficientNet_B7':
        model = DSIFN_EfficientNet_B7(in_channel, out_channel, 
                             backbone_type='random',
                             finetune=True)
        model = model.to(device=device)  
        
    elif name == 'DSIFN_EfficientNet_B4':
        model = DSIFN_EfficientNet_B4(in_channel, out_channel, 
                                backbone_type='random',
                                finetune=True)
        model = model.to(device=device) 
        
    # elif name == 'DSIFN_EfficientNet_B2':
    #     model = DSIFN_EfficientNet_B2(in_channel, out_channel, 
    #                          backbone_type='random',
    #                          finetune=True)
    #     model = model.to(device=device)  
        
    return model
    
def get_criterion(name, weights=None):
    criterion = None

    if name == 'DiceLoss':
        criterion = DiceLoss()
    if name == 'CombineLoss':
        criterion = CombineLoss(weights=weights)
    if name == 'BCELoss':
        criterion = BCELoss(weights=weights)
    if name == 'DS_Loss':
        criterion = DS_Loss_all(weights=weights)

    return criterion

def get_optimizer(name, model):
    optimizer = None
    if name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4) 
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters())
    return optimizer

def get_scheduler(name, optimizer):
    scheduler = None
    if name == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    elif name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=15,
                                                    gamma=0.8)
    return scheduler

def save_as_textfile(path, history):
    with open(path, "w") as f:
        for s in history:
            f.write(str(s) +"\n")
            
def plot_history_and_save(id, name, train_history, val_history, res_dir):
    n_epochs = len(train_history)
    t = np.linspace(1, n_epochs, n_epochs)

    plt.figure(num=id)
    plt.clf()
    l1_1, = plt.plot(t, train_history, label='Train_'+name)
    l1_2, = plt.plot(t, val_history, label='Val_'+name)
    plt.legend(handles=[l1_1, l1_2])
    plt.title(name)
    save_name = name + '.png'
    save_name = os.path.join(res_dir,save_name)
    plt.savefig(save_name)
            
def save_history(res_dir, history):
    
    epoch_train_loss = history['train_loss'] 
    epoch_train_accuracy = history['train_accuracy'] 
    epoch_val_loss = history['val_loss'] 
    epoch_val_accuracy = history['val_accuracy']    
    
    #Loss
    id = 1
    name = 'Loss'
    plot_history_and_save(id, name, epoch_train_loss, epoch_val_loss, res_dir)
    
    save_as_textfile(os.path.join(res_dir,"val_loss_history.txt"),epoch_val_loss)
    save_as_textfile(os.path.join(res_dir,"train_loss_history.txt"),epoch_train_loss)

    #Accuracy
    
    id = 2
    name = 'Accuracy'
    plot_history_and_save(id, name, epoch_train_accuracy, epoch_val_accuracy, res_dir)
    
    save_as_textfile(os.path.join(res_dir,"val_accuracy_history.txt"),epoch_val_accuracy)
    save_as_textfile(os.path.join(res_dir,"train_accuracy_history.txt"),epoch_train_accuracy)
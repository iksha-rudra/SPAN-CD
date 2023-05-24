#%%
DISC = 'Bulding model with pretrained resnet with s2mtcp, no fine tuning'

import torch
import random
import numpy as np
import os
import torch.nn as nn
from time import time, ctime
import matplotlib.pyplot as plt
import numpy as np

from train import train
from dataloaders import *
from predict import predict_and_save_test_results
from metrics import check_accuracy

from train_utils import *
from test_predict import test

# MODEL_NAME = 'DSIFN_ResNet'
# MODEL_NAME = 'DSIFN_EfficientNet_B7'
# MODEL_NAME = 'DSIFN_EfficientNet_B1'
# MODEL_NAME = 'DSIFN_EfficientNet_B0'
# MODEL_NAME = 'DSIFN_EfficientNet_B4'
# MODEL_NAME = 'SiamSegment'
# MODEL_NAME = 'ResUNet'
# MODEL_NAME = 'FC_EF'
# MODEL_NAME = 'SiamUNet_conc'
# MODEL_NAME = 'SiamUNet_diff'
# MODEL_NAME = 'UCDNet'
# MODEL_NAME = 'RDPNet'
# MODEL_NAME = 'BIT'
# MODEL_NAME = 'LUNet'
MODEL_NAME = 'DTCDSDN'
# MODEL_NAME = 'CDNet'
# MODEL_NAME = 'ChangeFormer'
# MODEL_NAME = 'SwinYNet'    

BANDS = ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
# BANDS = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
# BANDS = ['B03','B04','B8A', 'B11', 'B12']
# BANDS = ['B02','B03','B04']

PATH_TO_TRAIN_DATA = '../../../../DataSet/OSCD/'
PATH_TO_TEST_DATA = '../../../../DataSet/OSCD/'
PATH_TO_VAL_DATA = '../../../../DataSet/OSCD/'
PATH_TO_PRED = '../../../../DataSet/pred'

NUM_EPOCHS = 50
BATCH_SIZE = 8
PATCH_SIDE = 256
STRIDE = 128
OPTIMIZER_NAME = 'AdamW'
LOSS_FN_NAME = 'CombineLoss' 
# LOSS_FN_NAME = 'DS_Loss'
FP_MODIFIER = 10
SCHEDULER_NAME ='StepLR'
    
#Final training

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def do_train(model_name, 
             bands,
             path_to_train_data,
             path_to_test_data,
             path_to_val_data,                          
             path_to_pred,
             num_epochs,
             batch_size,
             patch_side,
             stride,
             optimizer_name,
             loss_fn_name,
             fp_modifier,
             scheduler_name):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    res_dir = create_result_directory(path_to_pred)
    
    model = get_model(model_name,len(bands), 2)
    
    train_loader, weights = train_dataloader(path=path_to_train_data,
                                                   batch_size=batch_size,
                                                   patch_side=patch_side,
                                                   stride=stride,
                                                   shuffle=False,
                                                   bands=bands)
    
    val_loader, val_dataset = val_dataloader(path=path_to_val_data,
                                batch_size=batch_size,
                                patch_side=patch_side,
                                stride=stride,
                                shuffle=True,
                                bands=bands)
    
    test_loader, test_dataset = test_dataloader(path=path_to_test_data,
                                batch_size=batch_size,
                                patch_side=patch_side,
                                stride=stride,
                                shuffle=False,
                                bands=bands)
    
    print('testdset:',len(test_dataset))
    
    weights = torch.FloatTensor(weights).to(device=device)
    
    criterion = get_criterion(loss_fn_name, weights)
    
    optimizer = get_optimizer(optimizer_name, model)
    
    scheduler = get_scheduler(optimizer=optimizer,
                              name=scheduler_name)
    
    t_start = time()
    print(f'Starting Time: {ctime(t_start)}')
    
    history = train(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler)
    
    t_end = time()
    
    print(f'Ending Time: {ctime(t_end)}')
    
    save_history( res_dir=res_dir, history=history )
    
    save_str = 'model.pth.tar'
    save_str = os.path.join(res_dir,save_str)
    torch.save(model.state_dict(), save_str)
    
    # predict_and_save_test_results(model, test_dataset, res_dir)
    metrics_list = test(dset=test_dataset, criterion=criterion, res_dir=res_dir, net=model)
    print(metrics_list)
    save_as_textfile(os.path.join(res_dir,"validation_result.txt"),metrics_list)
    
    #save hyper parameters
    validate_list = []

    validate_list.append(f'model_name: {model_name}')
    validate_list.append(f'bands: {bands}')
    validate_list.append(f'num_epochs: {num_epochs}')
    validate_list.append(f'batch_size: {batch_size}')
    validate_list.append(f'patch_side: {patch_side}')
    validate_list.append(f'stride: {stride}')
    validate_list.append(f'optimizer_name: {optimizer_name}')
    validate_list.append(f'loss_fn_name: {loss_fn_name}')
    validate_list.append(f'fp_modifier: {fp_modifier}')
    validate_list.append(f'scheduler_name: {scheduler_name}') 
    validate_list.append(f'Description: {DISC}')   

    print(validate_list)
    save_as_textfile(path=os.path.join(res_dir,"hyper_parameter_result.txt"),history=validate_list)
    
if __name__ == '__main__':

    do_train(model_name = MODEL_NAME,
             bands = BANDS,
             path_to_train_data = PATH_TO_TRAIN_DATA,
             path_to_test_data = PATH_TO_TEST_DATA,
             path_to_val_data = PATH_TO_VAL_DATA,
             path_to_pred = PATH_TO_PRED,
             num_epochs = NUM_EPOCHS,
             batch_size = BATCH_SIZE,
             patch_side = PATCH_SIDE,
             stride = STRIDE,
             optimizer_name = OPTIMIZER_NAME,
             loss_fn_name = LOSS_FN_NAME,
             fp_modifier = FP_MODIFIER,
             scheduler_name = SCHEDULER_NAME)

# %%

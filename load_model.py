import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm as tqdm
from torch.autograd import Variable
import numpy as np
import matplotlib.image as mpimg

from train_utils import get_model
from dataset.oscd_dataset import OSCD_Dataset

PATH_TO_PRED = '../../../../DataSet/pred/Mar-14-2023/11_26_05'
PATH_TO_TEST_DATA = '../../../../DataSet/OSCD/'
MODEL_NAME = 'DSIFN_ResNet'
BANDS = ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12']

def do_load_pred_save():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dset = OSCD_Dataset(path=PATH_TO_TEST_DATA,
                                fname='test.txt',
                                patch_side=256,
                                stride=128,
                                normalize=True,
                                transform=None,
                                bands=BANDS
                                )

    model = get_model(MODEL_NAME,len(BANDS), 2)
    model.load_state_dict(torch.load(os.path.join(PATH_TO_PRED, 'model.pth.tar'), map_location=torch.device(device=device)))
    print('LOAD OK')
    
    for img_index in tqdm(range(len(dset))):

        sample = dset[img_index]
        I1_full = sample['I1']
        I2_full = sample['I2']
        cm_full = sample['label']
        fname = sample['fname']
    
        I1 = Variable(torch.unsqueeze(I1_full, 0).float()).to(device)
        I2 = Variable(torch.unsqueeze(I2_full, 0).float()).to(device)
    
        out = model(I1.float(), I2.float())
        
        _, predicted = torch.max(out.data, 1)
        pr = (predicted.int() > 0).cpu().numpy()
        gt = (cm_full.data.int() > 0).cpu().numpy()

        #For Saving image
        predicted = torch.squeeze(predicted)
        cm = torch.squeeze(cm_full)
        cm = cm.type(torch.uint8)
        predicted = predicted.type(torch.uint8)
        
        I = np.stack((255*cm,255*np.squeeze(predicted.cpu().numpy()),255*cm),2)
        # I = np.stack((255*np.squeeze(predicted.cpu().numpy()),
        #                 255*np.squeeze(predicted.cpu().numpy()),
        #                 255*np.squeeze(predicted.cpu().numpy())),2)
        
        pred_fname = os.path.splitext(fname)[0]+'-'+str(img_index)+'-predicted.png'
        pred_fname = os.path.join(PATH_TO_PRED, pred_fname)
        mpimg.imsave(pred_fname,I.astype(np.uint8))

if __name__ == '__main__':
    do_load_pred_save()

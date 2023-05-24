#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from time import time, ctime
import matplotlib.pyplot as plt
import os

from ssl_loader import get_s2mtcp_ssl_loader
from ssl_loader import get_oscd_ssl_loader
from ssl_loader import get_sen12ms_ssl_loader
from ssl_train import train

# PATH_TO_DATASET = '../../../../DataSet/S2MTCP/'

# PATH_TO_DATASET = '../../../../DataSet/OSCD/'
PATH_TO_DATASET = ['/home/rakesh/DataSet/SEN12MS/ROIs1868_summer/',
                   '/home/rakesh/DataSet/SEN12MS/ROIs1158_spring',
                    '/home/rakesh/DataSet/SEN12MS/ROIs1970_fall',
                    '/home/rakesh/DataSet/SEN12MS/ROIs2017_winter']

PATCH_SIDE = 256
TRAIN_STRIDE = int(PATCH_SIDE/2)
BATCH_SIZE = 256
BANDS = ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
NUM_EPOCHS = 200

RES_DIR = '../../../loss_history'

# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

def create_ssl_result_directory(path):
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

def save_as_textfile(path, history):
    with open(path, "w") as f:
        for s in history:
            f.write(str(s) +"\n")

def plot_history_and_save(id, name, train_history, res_dir,start,end):
    n_epochs = len(train_history)
    t = np.linspace(1, n_epochs, n_epochs)

    plt.figure(num=id)
    plt.clf()
    l1_1, = plt.plot(t, train_history, label='Train_'+name)
    plt.legend(handles=[l1_1])
    plt.title(name)
    save_name = (f"train_loss_history_{start}_{end}.png")
    save_name = os.path.join(res_dir,save_name)
    plt.savefig(save_name)
    file_name = (f"train_loss_history_{start}_{end}.txt")
    save_as_textfile(os.path.join(res_dir,file_name), train_history)

def do_train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # train_loader = get_oscd_ssl_loader(PATH_TO_DATASET,
    #                                 patch_side=PATCH_SIDE,
    #                                 stride=TRAIN_STRIDE,
    #                                 batch_size=BATCH_SIZE)

    # train_loader = get_s2mtcp_ssl_loader(path=PATH_TO_DATASET,
    #                                         patch_side=PATCH_SIDE,
    #                                         stride=TRAIN_STRIDE,
    #                                         batch_size=BATCH_SIZE,
    #                                         bands=BANDS)
    
    train_loader = get_sen12ms_ssl_loader(path=PATH_TO_DATASET,
                                        patch_side=PATCH_SIDE,
                                        stride=TRAIN_STRIDE,
                                        batch_size=BATCH_SIZE)

    t_start = time()
    print(f'Starting Time: {ctime(t_start)}')
    
    history, start, end = train(train_loader, epochs=NUM_EPOCHS)

    # res_dir = create_ssl_result_directory(RES_DIR)
    if os.path.exists(RES_DIR) and os.path.isdir(RES_DIR):
        print("my_directory exists!")
    else:
        os.makedirs(RES_DIR) 
        
    plot_history_and_save(1, name='Train Loss', 
                          train_history=history, 
                          res_dir=RES_DIR,
                          start=start,
                          end=end)
    
    t_end = time()
    print(f'Ending Time: {ctime(t_end)}')

if __name__ == '__main__':
    do_train()
# %%

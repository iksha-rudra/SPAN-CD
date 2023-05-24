import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time as time
import math
import sys
import json

# from torch.utils.tensorboard import SummaryWriter
from model.barlow_twins import BarlowTwins, LARS

from tqdm import tqdm as tqdm

RESUME = False
LAST_FINAL_EPOCH = 4
LR = 0.2
COS = False
SCHEDULE = [120,160]
LEARNING_RATE_WEIGHTS = 0.2
LEARNING_RATE_BIASES = 0.0048
RANK = 0
CHECK_POINT_DIR = '../../../checkpoints'
PRINT_FREQ = 10
WEIGHT_DECAY=1e-4

def adjust_learning_rate(lr, cos, schedule, learning_rate_weights, learning_rate_biases, optimizer, epoch, epochs):
    w = 1
    if cos:
        w *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    else:
        for milestone in schedule:
            w *= 0.1 if epoch >= milestone else 1.
    optimizer.param_groups[0]['lr'] = w * learning_rate_weights
    optimizer.param_groups[1]['lr'] = w * learning_rate_biases

def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def train(train_loader, epochs):
    t = np.linspace(1, epochs, epochs)
    epoch_train_loss = []
    train_loss = 0.0

    # create tb_writer
    if RANK==0 and not os.path.isdir(CHECK_POINT_DIR):
        os.mkdir(CHECK_POINT_DIR)
    # if RANK==0:
    #     tb_writer = SummaryWriter(os.path.join(CHECK_POINT_DIR,'log'))

    if RANK == 0:

        stats_file = open(os.path.join(CHECK_POINT_DIR,  'stats.txt'), 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    fix_random_seeds(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = BarlowTwins().to(device=device)
    
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    
    optimizer = LARS(parameters, lr=0, weight_decay=WEIGHT_DECAY,
                    weight_decay_filter=True,
                    lars_adaptation_filter=True)
    
    # automatically resume from checkpoint if it exists
    checkpoint_file = os.path.join(CHECK_POINT_DIR, 'checkpoint_{:04d}.pth'.format(LAST_FINAL_EPOCH))
    start_epoch = 0

    if os.path.isfile(checkpoint_file) and RESUME:
        ckpt = torch.load(checkpoint_file,
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    
    print('Start training...')

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, epochs):

        adjust_learning_rate(lr=LR, cos=COS,
                            schedule=SCHEDULE, learning_rate_biases=LEARNING_RATE_BIASES,
                            learning_rate_weights=LEARNING_RATE_WEIGHTS,
                            optimizer=optimizer, epochs=epochs, epoch=epoch)
        
        loop = tqdm(enumerate(train_loader, start=epoch * len(train_loader)), 
                    total=len(train_loader),
                    leave=True)
                
        for step, (y1, y2) in loop:
            y1 = y1.float().to(device=device)
            y2 = y2.float().to(device=device)
            
            #adjust_learning_rate(args, optimizer, train_loader, step)
            optimizer.zero_grad()
            
            # with torch.cuda.amp.autocast():
            loss, on_diag, off_diag = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                lss = loss.item()
                train_loss += lss
            
            #update progress bar
            loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
            loop.set_postfix(train_loss = loss.item())

            if step % PRINT_FREQ == 0:
                if RANK == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 #lr=optimizer.param_groups['lr'],
                                 loss=loss.item(),
                                 on_diag=on_diag.item(),
                                 off_diag=off_diag.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)

        if RANK == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                        optimizer=optimizer.state_dict())
            torch.save(state, os.path.join(CHECK_POINT_DIR, 'checkpoint_{:04d}.pth'.format(epoch)))
        
        train_loss = train_loss / len(train_loader)
        epoch_train_loss.append(train_loss)

            
    if RANK == 0:
        # save final model
        torch.save(model.backbone.state_dict(),
                os.path.join(CHECK_POINT_DIR, 'checkpoint.pth'))
    
    return epoch_train_loss, start_epoch, epochs
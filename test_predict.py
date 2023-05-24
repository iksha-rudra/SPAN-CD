from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm as tqdm
from math import ceil
import numpy as np
from skimage import io
import os
import cv2
import matplotlib.image as mpimg

DS = False

def generate_predicted_output(out, type='argmax'):
    
    if type == 'argmax':

        if DS:
            _, predicted = torch.max(out[0].data, 1)
        else:
            _, predicted = torch.max(out.data, 1)
            
        pr = (predicted.int() > 0).cpu().numpy()
    
    elif type == 'otsu':

        sigmo = torch.nn.Sigmoid()
        

        arr = out.cpu().detach().numpy()



        blur = cv2.GaussianBlur(arr,(3,3),0)
        ret3,pr = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        pr = cv2.dilate(pr, kernel, iterations=1)
    
    return pr

def test(dset, net, res_dir, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smooth = 1e-5
    net.eval()
    
    n = 2
    tp = 0
    
    tp = np.float64(tp)
    tn = np.float64(tp)
    fp = np.float64(tp)
    fn = np.float64(tp)

    tot_count = np.float64(tp)
    tot_loss = np.float64(tp)
    
    for img_index in tqdm(range(len(dset))):
        sample = dset[img_index]
        I1 = sample['I1']
        I2 = sample['I2']
        cm = sample['label']
        fname = sample['fname']

        I1 = Variable(torch.unsqueeze(I1, 0).float()).to(device)
        I2 = Variable(torch.unsqueeze(I2, 0).float()).to(device)
        cm = Variable(cm).to(device=device)

        output = net(I1.float(), I2.float())
        
        # loss = criterion(output, cm.long())
        if DS:
            label_list = sample['list']
            loss = criterion(output, label_list)
        else:
            loss = criterion(output, cm.long())
        
        tot_loss += loss.data * np.prod(cm.size())
            
        tot_count += np.prod(cm.size())

        # _, predicted = torch.max(output.data, 1)

        if DS:
            _, predicted = torch.max(output[0].data, 1)
        else:
            _, predicted = torch.max(output.data, 1)

        pr = generate_predicted_output(output, 'argmax')
        gt = (cm.data.int() > 0).cpu().numpy()
        
        tp += np.logical_and(pr, gt).sum()
        tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
        fp += np.logical_and(pr, np.logical_not(gt)).sum()
        fn += np.logical_and(np.logical_not(pr), gt).sum()
        
        #For Saving image
        predicted = torch.squeeze(predicted)
        cm = torch.squeeze(cm)
        cm = cm.type(torch.uint8)
        predicted = predicted.type(torch.uint8)
        
        # I = np.stack((255*cm,255*np.squeeze(predicted.cpu().numpy()),255*cm),2)
        I = np.stack((255*np.squeeze(predicted.cpu().numpy()),
                        255*np.squeeze(predicted.cpu().numpy()),
                        255*np.squeeze(predicted.cpu().numpy())),2)
        
        pred_fname = os.path.splitext(fname)[0]+'-'+str(img_index)+'-predicted.png'
        pred_fname = os.path.join(res_dir, pred_fname)
        mpimg.imsave(pred_fname,I.astype(np.uint8))
        
    net_loss = tot_loss/tot_count        
    net_loss = float(net_loss.cpu().numpy())

    prec_nc = tn / (tn + fn)
    rec_nc = tn / (tn + fp)
    
    net_accuracy = (tp + tn + smooth)/(tot_count + smooth)
    prec = (tp) / (tp + fp + smooth)
    rec = (tp) / (tp + fn + smooth)
    f_meas = (2 * prec * rec ) / (prec + rec + smooth)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    N = tp + tn + fp + fn
    p0 = (tp + tn + smooth) / (N + smooth)
    pe = ( ( tp + fp  ) * ( tp + fn ) + ( tn + fp ) * ( tn + fn ) + smooth ) / (N * N + smooth)
    kappa = (p0 - pe + smooth) / (1 - pe + smooth)
    jaccard = (tp + smooth) / (tp + fp + fn + smooth)
  
    metrics_list = []
    metrics_list.append(f'net_loss: {net_loss}')
    metrics_list.append(f'net_accuracy: {net_accuracy}')
    metrics_list.append(f'precision: {prec}')
    metrics_list.append(f'recall: {rec}')
    metrics_list.append(f'dice: {dice}')
    metrics_list.append(f'f_meas: {f_meas}')    
    metrics_list.append(f'kappa: {kappa}')
    metrics_list.append(f'jaccard: {jaccard}')
    metrics_list.append(f'prec_nc : {prec_nc}')
    metrics_list.append(f'rec_nc : {rec_nc}')
    
    return metrics_list
    
def save_test_results(dset, net, path='', net_name='model-name'):
    for name in tqdm(dset.names):
            I1, I2, cm = dset.get_img(name)
            I1 = Variable(torch.unsqueeze(I1, 0).float()).cuda()
            I2 = Variable(torch.unsqueeze(I2, 0).float()).cuda()
            out = net(I1, I2)
            _, predicted = torch.max(out.data, 1)
            I = np.stack((255*cm,255*np.squeeze(predicted.cpu().numpy()),255*cm),2)
            file = (f'{net_name}-{name}.png',I)
            io.imsave(os.path.join(path, file))
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

L = 1024

def generate_predicted_output(out, type='argmax'):
    
    if type == 'argmax':
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

def test(dset, net, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smooth = 1e-5
    net.eval()
    
    n = 2
    class_correct = list(0. for i in range(n))
    class_total = list(0. for i in range(n))
    class_accuracy = list(0. for i in range(n))
    
    tp = 0
    
    tp = np.float64(tp)
    tn = np.float64(tp)
    fp = np.float64(tp)
    fn = np.float64(tp)

    tot_count = np.float64(tp)
    tot_loss = np.float64(tp)
    
    for img_index in tqdm(range(len(dset))):
        sample = dset[img_index]
        I1_full = sample['I1']
        I2_full = sample['I2']
        cm_full = sample['label']
        
        s = cm_full.shape
        
        for ii in range(ceil(s[0]/L)):
            for jj in range(ceil(s[1]/L)):
                xmin = L*ii
                xmax = min(L*(ii+1),s[1])
                ymin = L*jj
                ymax = min(L*(jj+1),s[1])
                I1 = I1_full[:, xmin:xmax, ymin:ymax].to(device)
                I2 = I2_full[:, xmin:xmax, ymin:ymax].to(device)
                cm = cm_full[xmin:xmax, ymin:ymax].to(device)

                I1 = Variable(torch.unsqueeze(I1, 0).float()).to(device)
                I2 = Variable(torch.unsqueeze(I2, 0).float()).to(device)
                # cm = Variable(torch.unsqueeze(1.0*cm,0).float()).to(device)
                # cm = np.einsum('ijk->jki',cm)

                output = net(I1.float(), I2.float())
                
                loss = criterion(output, cm.long())
                tot_loss += loss.data * np.prod(cm.size())
                    
                tot_count += np.prod(cm.size())

                _, predicted = torch.max(output.data, 1)

                c = (predicted.int() == cm.data.int())
                for i in range(c.size(1)):
                    for j in range(c.size(2)):
                        l = int(cm.data[0, i, j])
                        class_correct[l] += c[0, i, j]
                        class_total[l] += 1
                        
                # pr = (predicted.int() > 0).cpu().numpy()

                pr = generate_predicted_output(output, 'argmax')
                gt = (cm.data.int() > 0).cpu().numpy()
                
                tp += np.logical_and(pr, gt).sum()
                tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                fp += np.logical_and(pr, np.logical_not(gt)).sum()
                fn += np.logical_and(np.logical_not(pr), gt).sum()
        
    net_loss = tot_loss/tot_count        
    net_loss = float(net_loss.cpu().numpy())
    
    net_accuracy = 100 * (tp + tn)/tot_count
    
    for i in range(n):
        class_accuracy[i] = 100 * class_correct[i] / max(class_total[i],0.00001)
        class_accuracy[i] =  float(class_accuracy[i].cpu().numpy())

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
    metrics_list.append(f'class_accuracy: {class_accuracy}')
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
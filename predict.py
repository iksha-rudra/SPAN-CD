import torch
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.image as mpimg
from tqdm import tqdm as tqdm
import cv2

from train_utils import save_as_textfile

def generate_predicted_output(out, type='argmax'):
    
    if type == 'argmax':
        _, predicted = torch.max(out.data, 1)
        pr = (predicted.int() > 0).cpu().numpy()
    
    elif type == 'otsu':

        arr = out.cpu().numpy()

        blur = cv2.GaussianBlur(arr,(3,3),0)
        ret3,pr = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        pr = cv2.dilate(pr, kernel, iterations=1)
    
    return pr

def predict_and_save_test_results(model, dset, path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smooth = 1e-5
    
    tot_loss = np.float64(0)
    tot_count = np.float64(0)
    
    tp = np.float64(0)
    tn = np.float64(0)
    fp = np.float64(0)
    fn = np.float64(0)
    
    for img_index in tqdm(dset.names):
        I1_full, I2_full, cm_full = dset.get_img(img_index)
        
        with torch.no_grad():
            I1 = Variable(torch.unsqueeze(I1, 0).float().to(device=device))
            I2 = Variable(torch.unsqueeze(I2, 0).float().to(device=device))
            out = model(I1, I2)
            
        #To evaluate
        pr = generate_predicted_output(out, 'argmax')
        gt = (cm_full.data.int() > 0).cpu().numpy()
        
        '''
        #For Saving image
        predicted = torch.squeeze(predicted)
        cm = torch.squeeze(cm)
        cm = cm.type(torch.uint8)
        predicted = predicted.type(torch.uint8)
        
        # I = np.stack((255*cm,255*np.squeeze(predicted.cpu().numpy()),255*cm),2)
        I = np.stack((255*np.squeeze(predicted.cpu().numpy()),
                      255*np.squeeze(predicted.cpu().numpy()),
                      255*np.squeeze(predicted.cpu().numpy())),2)
        
        pred_fname = os.path.splitext(fname)[0]+'-'+str(i)+'-predicted.png'
        pred_fname = os.path.join(path,pred_fname)
        mpimg.imsave(pred_fname,I)
        '''
        
        tot_count += np.prod(cm.size())
        
        tp += np.logical_and(pr, gt).sum()
        tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
        fp += np.logical_and(pr, np.logical_not(gt)).sum()
        fn += np.logical_and(np.logical_not(pr), gt).sum()
        
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
    metrics_list.append(f'Test Accuracy: {net_accuracy}')
    metrics_list.append(f'Test Precision: {prec}')
    metrics_list.append(f'Test Recall: {rec}')
    metrics_list.append(f'Test F1-Score: {f_meas}')
    metrics_list.append(f'Test Kappa: {kappa}')
    metrics_list.append(f'Test Jaccard: {jaccard}')
    metrics_list.append(f'Test Dice: {dice}')

    print(metrics_list)
    save_as_textfile(os.path.join(path,"validation_result.txt"),metrics_list)
        
        
from torch.autograd import Variable
import torch
import numpy as np
from tqdm import tqdm as tqdm

def check_accuracy(loader, model, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    tot_loss = np.float16(0)
    tot_count = 0
    
    tp = np.float16(0)
    tn = np.float16(0)
    fp = np.float16(0)
    fn = np.float16(0)
     
    for i,batch in enumerate(tqdm(loader)):
    
        with torch.no_grad():
            I1 = Variable(batch['I1'].float().to(device=device))
            I2 = Variable(batch['I2'].float().to(device=device))

            cm = Variable(batch['label'].to(device=device))
            cm = torch.squeeze(cm,dim=1)
        
            output = model(I1, I2)
            
            loss = criterion(output, cm.long())
            
        tot_loss += loss.data * np.prod(cm.size())
        tot_count += np.prod(cm.size())

        _, predicted = torch.max(output.data, 1)
                
        pr = (predicted.int() > 0).cpu().numpy()
        gt = (cm.data.int() > 0).cpu().numpy()
        
        tp += np.logical_and(pr, gt).sum()
        tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
        fp += np.logical_and(pr, np.logical_not(gt)).sum()
        fn += np.logical_and(np.logical_not(pr), gt).sum()

    total_loss = tot_loss.type(torch.DoubleTensor)
    tot_count = np.float16(tot_count)

    net_loss = total_loss/tot_count
    net_loss = np.float(net_loss.cpu().numpy())
    net_accuracy = (tp + tn)/tot_count
    
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f_meas = 2 * prec * rec / (prec + rec)
    
    N = tp + tn + fp + fn
    
    p0 = (tp + tn) / N
    pe = ((tp+fp)*(tp+fn) + (tn+fp)*(tn+fn)) / (N * N)
    
    kappa = (p0 - pe) / (1 - pe)
    jaccard = (tp) / (tp + fp + fn) 
    
    metrics = [net_accuracy,net_loss, prec, rec, f_meas, kappa, jaccard]
        
    return metrics
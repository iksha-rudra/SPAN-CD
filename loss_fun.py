import torch
import torch.nn as nn
import numpy as np

class WBCCE(nn.Module):
    def __init__(self):
        super(WBCCE, self).__init__()
    
    def forward(self, lsm_vector, targets):         #inputs in LogSoftmax

        tot_lss = 0
        
        batch_size = lsm_vector.shape[0]
        
        for i in range(batch_size):
    
            lsm_v1 = lsm_vector[i][0]
            lsm_v2 = lsm_vector[i][1]
            
            target = targets[i]
            
            s = target.shape
            
            n_pix = float(np.prod(s))
            true_pix = float(target.sum())
            
            p_weight = 10 * 2 * true_pix / n_pix
            n_weight = 2 * (n_pix - true_pix) / n_pix
            
            neg_target = 1 - target
            
            neg_target = neg_target * p_weight 
            target = target * n_weight
                
            sum = neg_target.sum() + target.sum()
            
            sum_mat = torch.stack((lsm_v1 * neg_target, lsm_v2 * target)).sum(0)
            lss = -sum_mat.sum()
            
            lss = lss / sum
            
            tot_lss += lss
        
        return tot_lss
    
class MyFocalLoss(nn.Module):
    def __init__(self, gamma):
        super(MyFocalLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, lsm_vector, targets):         #inputs in LogSoftmax
        
        tot_lss = 0
        
        batch_size = lsm_vector.shape[0]
        
        sm_vector = torch.exp(lsm_vector)
        sm_vector = sm_vector ** self.gamma
        
        for i in range(batch_size):
            
            sm_v1 = sm_vector[i][0]
            sm_v2 = sm_vector[i][1]
            
            lsm_v1 = lsm_vector[i][0]
            lsm_v2 = lsm_vector[i][1]
            
            target = targets[i]
            
            s = target.shape    
            
            n_pix = float(np.prod(s))
            true_pix = float(target.sum())
            
            p_weight = 10 * 2 * true_pix / n_pix
            n_weight = 2 * (n_pix - true_pix) / n_pix
            
            neg_target = 1 - target
            
            neg_target = neg_target * p_weight 
            target = target * n_weight
                
            sum = neg_target.sum() + target.sum()
            
            sum_mat = torch.stack((lsm_v1 * neg_target * sm_v2, lsm_v2 * target * sm_v1)).sum(0)
            lss = -sum_mat.sum()
            
            lss = lss / sum
            
            tot_lss += lss
    
        return tot_lss

class LogCosh_Kappa(nn.Module):
    """Some Information about LogCosh_Kappa"""
    def __init__(self, alpha=0.5, beta=0.5):
        super(LogCosh_Kappa, self).__init__()

    def forward(self, lsm_vector, targets):         #inputs in LogSoftmax

        tot_lss = 0
        
        batch_size = lsm_vector.shape[0]
        
        for i in range(batch_size):
    
            lsm_v1 = lsm_vector[i][0]
            lsm_v2 = lsm_vector[i][1]
            
            target = targets[i]
            
            s = target.shape
            
            n_pix = float(np.prod(s))
            true_pix = float(target.sum())
            
            p_weight = 10 * 2 * true_pix / n_pix
            n_weight = 2 * (n_pix - true_pix) / n_pix
            
            neg_target = 1 - target
            
            neg_target = neg_target * p_weight 
            target = target * n_weight
                
            sum = neg_target.sum() + target.sum()
            
            sum_mat = torch.stack((lsm_v1 * neg_target, lsm_v2 * target)).sum(0)
            lss = -sum_mat.sum()
            
            lss = lss / sum
            
            tot_lss += lss
        
        return tot_lss

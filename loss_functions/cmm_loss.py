import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def CMM(T1,T2):
    
    _, M, N = T1.shape
    
    cmm = torch.zeros(M,N)
    cmm_dir = torch.zeros(M,N)

    for i in range(M):
        for j in range(N):
            tensor1 = T1[:,i,j]
            tensor2 = T2[:,i,j]
            dist = (tensor1 - tensor2).pow(2).sum(0).sqrt()
            cmm[i][j] = dist
            
            vec_1_n = tensor1.norm(p=2, dim=-1, keepdim=True)
            vec_2_n = tensor2.norm(p=2, dim=-1, keepdim=True)
            vec1 = tensor1 / vec_1_n
            vec2 = tensor2 / vec_2_n
            cos_ij = torch.acos(torch.dot(vec1, vec2))
            cmm_dir[i][j] = cos_ij
            
    cmm_norm = (cmm / cmm.max())
    cmm_norm_mean = cmm_norm.mean()
    cmm_dir_norm = (cmm_dir / cmm_dir.max())
    cmm_dir_norm_mean = cmm_dir_norm.mean()
    
    cmm_norm = torch.clip(cmm_norm, min=cmm_norm_mean)
    cmm_dir_norm = torch.clip(cmm_dir_norm, min=cmm_dir_norm_mean)

    return cmm_norm, cmm_dir_norm
import matplotlib.image as mpimg
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from math import floor, ceil, sqrt, exp
import random

NVDI = lambda b_nir, b_red: ( b_nir - b_red ) / ( b_nir + b_red )
NDWI = lambda b_nir, b_green: ( b_green - b_nir ) / ( b_green + b_nir )
SAVI = lambda b_nir, b_red, l: (( b_nir - b_red ) * (1 + l)) / ( b_nir + b_red + l)
NDWI = lambda b_green, b_nir: ( b_green - b_nir ) / ( b_green + b_nir )
MNDWI = lambda b_green, b_swir: ( b_green - b_swir ) / ( b_green + b_swir )
NDBI = lambda b_swir, b_nir: ( b_swir - b_nir ) / ( b_swir + b_nir )
NBI = lambda b_red, b_swir, b_nir: (b_red * b_swir) / b_nir
BRBA = lambda b_read, b_swir: (b_read) / (b_swir)
NBAI = lambda b_swir, b_swir2, b_green: (b_swir - b_swir2 / b_green) / (b_swir + b_swir2 / b_green)
MBI = lambda b_swir2, b_red, b_nir: (b_swir2 * b_red - b_nir ** 2) / (b_red + b_nir + b_swir2)
BAEI = lambda b_red, l, b_green, b_swir: (b_red + l) / (b_green + b_swir)

def IBI(b_swir, b_nir, b_red, b_green):
   return ( NDBI(b_swir, b_nir) - (SAVI(b_nir, b_red) + MNDWI(b_green, b_swir)) / 2 ) / \
          ( NDBI(b_swir, b_nir) + (SAVI(b_nir, b_red) + MNDWI(b_green, b_swir)) / 2 )

'''
def PC1(S2_image):
  # compute PC1 transformations
  
    # Reshape the image data into a 2D array
  data = data.reshape((-1, data.shape[-1])).astype(np.float32)

  # Calculate the mean of each band and center the data
  mean = np.mean(data, axis=0)
  data_centered = data - mean

  # Compute the covariance matrix of the centered data
  covariance_matrix = np.cov(data_centered.T)

  # Compute the eigenvectors and eigenvalues of the covariance matrix
  eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

  # Select the eigenvector corresponding to the largest eigenvalue
  pc1_index = np.argmax(eigenvalues)
  pc1 = eigenvectors[:, pc1_index]

  # Multiply the centered data by the selected eigenvector to obtain the PC1 transformation
  pc1_transform = np.dot(data_centered, pc1)
  
  return pc1_transform 

def CBI(S2_image):
  # mask out the water regions using NDWI
  
  PC1 = PC1(S2_image)
  
  #b_green, b_nir, b_read
  return ((PC1 + NDWI())/2 - SAVI) / \
          ((PC1 + NDWI())/2 + SAVI)
'''  
  
    
def add_normalised_differnce_indexes(S2_image):
  #Append all the indexs
  # NDVI, SAVI, NDWI, NDBI, IBI, NBI, NDSI, BRBA, NBAI, BCI, MBI, BAEI, CBI
  
  # B03 = b_green
  # B04 = b_red
  # B8A = b_nir
  # B11 = b_swir2
  # B12 = b_swir
  
  pass


def TCT_Transform(S2_image):
  
  tct_matrix_t = torch.tensor([ [ 0.0356, -0.0635,  0.0649 ],  
                              [ 0.0822, -0.1128,  0.1363 ], 
                              [ 0.1360, -0.1680,  0.2802 ], 
                              [ 0.2611, -0.3480,  0.3072 ], 
                              [ 0.2964, -0.3303,  0.5288 ], 
                              [ 0.3338,  0.0852,  0.1379 ], 
                              [ 0.3877,  0.3302, -0.0001 ],
                              [ 0.3895,  0.3165, -0.0807 ],
                              [ 0.0949,  0.0467, -0.0302 ],
                              [ 0.0009, -0.0009,  0.0003 ],
                              [ 0.3882, -0.4578, -0.4064 ],
                              [ 0.1366, -0.4064, -0.5602 ],
                              [ 0.4750,  0.3625, -0.1389 ]
                              ])

  tct_matrix = torch.einsum('ij->ji',tct_matrix_t)
  S2_tct = torch.einsum('ij,jkl->ikl', tct_matrix, S2_image)
  
  return S2_tct

def unimodal_thresholding():
  #to be done
  pass

def kapoor_thresholding():
  #to be done
  pass
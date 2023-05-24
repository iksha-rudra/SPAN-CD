
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from ssl_loader import get_oscd_ssl_loader

# PATH_TO_DATASET = '../../../../DataSet/OSCD/'
# PATCH_SIDE = 256
# TRAIN_STRIDE = int(PATCH_SIDE/2)
# BATCH_SIZE = 1

# train_loader = get_oscd_ssl_loader(PATH_TO_DATASET,
#                                    patch_side=PATCH_SIDE,
#                                    stride=TRAIN_STRIDE,
#                                    batch_size=BATCH_SIZE)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from ssl_loader import get_s2mtcp_ssl_loader

# dataset1 = train_loader.dataset
# print(len(dataset1))

# I1, I2 = dataset1[0]

# import numpy as np
# def get_rgb_ndarray(image):

#     data = image.numpy()
    
#     r = data[2,:,:]
#     g = data[3,:,:]
#     b = data[4,:,:]

#     rgb = np.stack([r,g,b])
#     rgb = np.einsum('ijk->jki',rgb)

#     return rgb

# import matplotlib.pyplot as plt
# f, axarr = plt.subplots(nrows=1, ncols=2,figsize=(25, 10),dpi=80)
# axarr[0].imshow(get_rgb_ndarray(I1))
# axarr[1].imshow(get_rgb_ndarray(I2))
# plt.show()

#%%
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from ssl_loader import get_s2mtcp_ssl_loader

# PATH_TO_DATASET = '../../../../../DataSet/S2MTCP/'
# PATCH_SIDE = 256
# TRAIN_STRIDE = int(PATCH_SIDE/2)
# BATCH_SIZE = 1
# BANDS = ['B02','B03','B04']

# s2mtcp_train_loader = get_s2mtcp_ssl_loader(path=PATH_TO_DATASET,
#                                             patch_side=PATCH_SIDE,
#                                             stride=TRAIN_STRIDE,
#                                             batch_size=BATCH_SIZE,
#                                             bands=BANDS)

# dataset1 = s2mtcp_train_loader.dataset
# print(len(dataset1))

# I1, I2 = dataset1[0]

# import numpy as np
# def get_rgb_ndarray(image):

#     data = image.numpy()
    
#     r = data[1,:,:]
#     g = data[2,:,:]
#     b = data[3,:,:]

#     rgb = np.stack([r,g,b])
#     rgb = np.einsum('ijk->jki',rgb)

#     return rgb

# import matplotlib.pyplot as plt
# f, axarr = plt.subplots(nrows=1, ncols=2,figsize=(25, 10),dpi=80)
# axarr[0].imshow(get_rgb_ndarray(I1))
# axarr[1].imshow(get_rgb_ndarray(I2))
# plt.show()

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ssl_loader import get_sen12ms_ssl_loader

PATH_TO_DATASET = ['/home/rakesh/DataSet/SEN12MS/ROIs1868_summer/',
                   '/home/rakesh/DataSet/SEN12MS/ROIs1158_spring',
                    '/home/rakesh/DataSet/SEN12MS/ROIs1970_fall',
                    '/home/rakesh/DataSet/SEN12MS/ROIs2017_winter']

# PATH_TO_DATASET = ['/home/rakesh/DataSet/SEN12MS/ROIs1868_summer/']

PATCH_SIDE = 224
TRAIN_STRIDE = int(PATCH_SIDE/2)
BATCH_SIZE = 1
BANDS = ['B02','B03','B04']

sen12ms_train_loader = get_sen12ms_ssl_loader(path=PATH_TO_DATASET,
                                              batch_size=BATCH_SIZE,
                                              patch_side=PATCH_SIDE,
                                              stride=TRAIN_STRIDE)

dataset1 = sen12ms_train_loader.dataset
print(len(dataset1))

I1, I2 = dataset1[0]

import numpy as np
def get_rgb_ndarray(image):

    data = image.numpy()
    
    r = data[1,:,:]
    g = data[2,:,:]
    b = data[3,:,:]

    rgb = np.stack([r,g,b])
    rgb = np.einsum('ijk->jki',rgb)

    return rgb

import matplotlib.pyplot as plt
f, axarr = plt.subplots(nrows=1, ncols=2,figsize=(25, 10),dpi=80)
axarr[0].imshow(get_rgb_ndarray(I1))
axarr[1].imshow(get_rgb_ndarray(I2))
plt.show()

# %%

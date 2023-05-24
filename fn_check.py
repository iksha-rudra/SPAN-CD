# %%
from dataset.oscd_sr_dataset import OSCD_SR_Dataset
from dataset.oscd_dataset import OSCD_Dataset

import torchvision.transforms as tr

from augmentations import RandomFlip, \
    RandomRotation, \
    GaussianBlur, \
    ToGray, \
    RandomResizedCrop, \
    Normalize, \
    RandomNoise, \
    RandomGeomRotation, \
    Solarization, \
    BandSwap, \
    BandTranslation

# data_transform = tr.Compose([
#     RandomResizedCrop(p=0.5),
#     RandomRotation(p=0.5),
#     RandomFlip(p=0.5),
#     tr.RandomApply(
#         [GaussianBlur(p=0.3),
#             ToGray(p=0.3),
#             RandomNoise(p=0.3),
#             RandomGeomRotation(0.3)
#             ],
#         p=0.8
#     ),
#     Solarization(p=0.5),
#     BandSwap(p=0.5),
#     BandTranslation(p=0.5),
#     Normalize(p=1.0)
# ])

data_transform = tr.Compose([
    RandomRotation(p=0.0)
])

transform_rotate_flip = tr.Compose([
    RandomFlip(p=1.0),
    RandomRotation(p=1.0)
])

PATH_TO_DATASET = '../../../../DataSet/OSCD/'
PATCH_SIDE = 256
TRAIN_STRIDE = int(PATCH_SIDE / 2)
BANDS = ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09', 'B10', 'B11','B12']

dset = OSCD_Dataset(path=PATH_TO_DATASET,
                              fname='train.txt',
                              patch_side=PATCH_SIDE,
                              stride=PATCH_SIDE-32,
                              normalize=True,
                              transform=transform_rotate_flip,
                              bands=BANDS
                              )
print(len(dset))

sample = dset[10]

I1 = sample['I1']
I2 = sample['I2']
label = sample['label']
lbl_lst = sample['list']

import numpy as np

def get_rgb_ndarray(image):
    data = image.numpy()

    r = data[5, :, :] * 255
    g = data[6, :, :] * 255 
    b = data[7, :, :] * 255

    rgb = np.stack([r, g, b])
    rgb = np.einsum('ijk->jki', rgb)

    rgb = rgb.astype(np.uint8)

    return rgb

import matplotlib.pyplot as plt

f, axarr = plt.subplots(nrows=1, ncols=8, figsize=(25, 25), dpi=80)
axarr[0].imshow(get_rgb_ndarray(I1))
axarr[1].imshow(get_rgb_ndarray(I2))
# axarr[2].imshow(np.einsum('ijk->jki', lbl_lst[0].numpy()))
# axarr[3].imshow(np.einsum('ijk->jki', lbl_lst[1].numpy()))
# axarr[4].imshow(np.einsum('ijk->jki', lbl_lst[2].numpy()))
# axarr[5].imshow(np.einsum('ijk->jki', lbl_lst[3].numpy()))
# axarr[6].imshow(np.einsum('ijk->jki', lbl_lst[4].numpy()))
# axarr[7].imshow(np.einsum('ijk->jki', lbl_lst[5].numpy()))
plt.show()

# %%

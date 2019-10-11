"""
Determine the appropriate threshold for acceptability of a MRI patch
The background region should be minimal.

author: ashish gupta
email: ashishagupta@gmail.com
date: 10-03-2019
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import nibabel
import skimage
import random

patch_h = 56
patch_w = 56


mri_path_file = 'D:/Repos/MI-DQA/utils/train-cam-office-0.csv'
mri_paths = pd.read_csv(mri_path_file, header=None)
mri_paths.columns = ['path']
mri_paths = mri_paths['path'].tolist()

mri_path = random.choice(mri_paths)
    
vol = nibabel.load(mri_path)
vol = vol.get_fdata()
[img_h, img_w, img_d] = vol.shape
# drop the bottom 25% and top 10% of the slices
vol = vol[:, :, int(img_d // 4):int(9 * img_d // 10)]
[img_h, img_w, img_d] = vol.shape
d = int(random.randint(0, img_d - 1))
# slide a patch sized window across a slice
vol = vol[:, :, d]
vol_pad = np.zeros((img_h+patch_h, img_w+patch_w))
vol_pad[patch_h//2:img_h+patch_h//2,patch_w//2:img_w+patch_w//2] = vol
img_h, img_w = vol_pad.shape
n_row = img_h-patch_h
n_col = img_w-patch_w
int_val = np.zeros((n_row, n_col))
for i in range(0, n_col):
    for j in range(0, n_row):
        patch = vol_pad[j+patch_w//2:j+3*patch_w//2, i+patch_h//2:i+3*patch_h//2]
        patch_bg = patch<64
        if patch_bg.sum() < 0.075*patch_w*patch_h:
            plt.imshow(patch, cmap='gray')
            plt.show()
        patch_int = np.sum(patch)/(patch_w*patch_h)
        int_val[j, i] = patch_int

X = list(range(n_row))
Y = list(range(n_col))
X,Y = np.meshgrid(X,Y)
# Z = np.transpose(int_val)
Z = int_val
print(X.shape, Y.shape, Z.shape)

plt.subplot(211)
plt.imshow(vol, cmap='gray')
plt.subplot(212)
plt.imshow(Z, cmap='jet')
plt.show()


    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig.colorbar(surf)
    # plt.imshow(vol, cmap='gray')
    # plt.show()
    # plt.savefig('mri_path_threshold.png')

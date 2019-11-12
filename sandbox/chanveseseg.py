"""
chan vese segmentation for MRI
author: ashish gupta
email: ashishagupta@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import chan_vese


import os
import random
import pandas as pd
import math
import scipy.misc
import time
import copy
import skimage.transform
from skimage import color
from skimage import io
from skimage import img_as_float
import cv2
import seaborn as sns

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    import nibabel


# mri_path = 'D:/Datasets/ABIDE1/OLIN/sub-0050120/anat/sub-0050120_T1w.nii.gz'
mri_path = 'D:/Datasets/ABIDE1/PITT/sub-0050005/anat/sub-0050005_T1w.nii.gz'

nii = nibabel.load(mri_path)
nii = nii.get_fdata()
[img_h, img_w, img_d] = nii.shape
image = nii[:,:,127]

# Feel free to play around with the parameters to see how they impact the result
cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=50,
               dt=0.5, init_level_set="checkerboard", extended_output=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
ax[1].set_title(title, fontsize=12)

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)

fig.tight_layout()
plt.show()
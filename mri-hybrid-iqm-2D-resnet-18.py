"""
mri-hybrid-iqm-2D-resnet-18-abide-1
"""
# DL Library
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms
from torchvision import models


import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.misc
import time 
import copy

# Nifti I/O
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    import nibabel

train_csv = 'train.csv'
val_csv = 'val.csv'
n_epoch = 100000
patch_h = 56
patch_w = 56

checkpoint_dir = './checkpoints/'
ckpt_path = checkpoint_dir+'mri-hybrid-iqm-dqa-2d-resnet-18.pth'
perf_path = checkpoint_dir+'mri-hybrid-iqm-dqa-2d-resnet-18.perf'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



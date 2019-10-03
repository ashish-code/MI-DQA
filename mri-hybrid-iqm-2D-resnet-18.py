"""
mri-hybrid-iqm-2D-resnet-18-abide-1

author: ashish gupta
email: ashishagupta@gmail.com
date: 09-18-2019
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

# NIFTI I/O
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    import nibabel

train_csv = 'utils/train-hybrid-tesla.csv'
val_csv = 'utils/val-hybrid-tesla.csv'
test_csv = ''
n_epoch = 1000
patch_h = 56
patch_w = 56

checkpoint_dir = './checkpoints/'
ckpt_path = checkpoint_dir+'mri-hybrid-iqm-dqa-2d-resnet-18.pth'
perf_path = checkpoint_dir+'mri-hybrid-iqm-dqa-2d-resnet-18.perf'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# >> Dataset class >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class MRIData(Dataset):
    """
    Returns MRI patch, IQM for volume (not that patch) and label (based on MOS)
    """
    def __init__(self, phase='train'):
        self.phase = phase
        if self.phase == 'train':
            self.data_path = train_csv
        elif self.phase == 'val':
            self.data_path = val_csv
        elif self.phase == 'test':
            self.data_path = test_csv
        self.df = pd.read_csv(self.data_path)

    def __getitem__(self, idx):
        """
        :param index:
        :return: torch array (mri patch), torch array (IQMs), label
        """
        _row = self.df.iloc[idx, :]
        _label = int(_row['label'])
        _path = _row['mri_path']
        _iqm = _row[1:66]
        nii = nibabel.load(_path)
        nii = nii.get_fdata()
        [img_h, img_w, img_d] = nii.shape
        # drop the bottom 25% and top 10% of the slices
        nii = nii[:, :, int(img_d / 4):int(9 * img_d / 10)]
        [img_h, img_w, img_d] = nii.shape
        _patch_h = patch_h
        _patch_w = patch_w
        if img_h < patch_h:
            _patch_h = img_h
        if img_w < patch_w:
            _patch_w = img_w
        # extract random slice and random patch
        h_l = int(random.randint(0, img_h - _patch_h))
        h_u = int(h_l + _patch_h - 1)
        w_l = int(random.randint(0, img_w - _patch_w))
        w_u = int(w_l + _patch_w - 1)
        d = int(random.randint(0, img_d - 1))
        nii = nii[h_l:h_u, w_l:w_u, d]
        # resize
        nii = scipy.misc.imresize(nii, (224, 224))
        # to pytorch tensor
        nii = torch.tensor(nii)
        nii.unsqueeze_(0)
        nii = nii.repeat(3, 1, 1)
        _iqm = torch.tensor(_iqm)
        return nii, _iqm, _label

    def __len__(self):
        return len(self.df.index)
# << Dataset class <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# dataset = MRIData('train')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=True)
#
# for ibatch, (nii, iqm, label) in enumerate(dataloader):
#     print(ibatch, nii.shape, iqm.shape, label.shape)


def main():
    # If pre-trained model is saved, use it
    if not os.path.exists(ckpt_path):
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)
        epoch = 0
        perf = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    # use the checkpoint
    else:
        model_ft = models.resnet18(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

        checkpoint = torch.load(ckpt_path)
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        exp_lr_scheduler.load_state_dict(checkpoint['scheduler'])
        perf = torch.load(perf_path)
        # resolving CPU vs GPU issue for optimzer.cuda()
        for state in optimizer_ft.state.values():
            for k,v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    criterion = nn.CrossEntropyLoss()
    model_ft = model_ft.to(device)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, epoch, perf, num_epochs=n_epoch)


if __name__=='__main__':
    main()



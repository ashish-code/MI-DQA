"""
Evaluate the performance of a pre-trained model on hold-out validation and test set

author: ashish gupta
email: ashishagupta@gmail.com
date: 09-26-2019
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
import scipy.ndimage
import time
import copy

# NIFTI I/O
import nibabel

# path to MRI scans and expert label
if os.name == 'nt':
    train_csv = 'train-office.csv'
    val_csv = 'val-office.csv'
    test_csv = 'test-office.csv'
else:
    train_csv = 'train-tesla.csv'
    val_csv = 'val-tesla.csv'
    test_csv = 'test-tesla.csv'

patch_h = 56
patch_w = 56

checkpoint_dir = './checkpoints/'
# ToDo: checkpoint file will be user input argument
# ckpt_path = checkpoint_dir+'mri-dqa-2d-resnet-18-epoch-350.pth'
# ckpt_path = checkpoint_dir + 'mri-dqa-2d-resnet-18.pth'
# ckpt_path = checkpoint_dir + 'mri-dqa-2d-resnet-18-rot.pth'
ckpt_path = checkpoint_dir + 'mri-dqa-2d-resnet-18-rot-onbrain.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _get_acceptable(patch):
    [img_h, img_w, img_d] = patch.shape
    # extract random slice and random patch
    acceptable = False
    while not acceptable:
        h_l = int(random.randint(0, img_h - patch_h))
        h_u = int(h_l + patch_h - 1)
        w_l = int(random.randint(0, img_w - patch_w))
        w_u = int(w_l + patch_w - 1)
        d = int(random.randint(0, img_d - 1))
        patch_t = patch[h_l:h_u, w_l:w_u, d]
        # select patch if overlapping sufficient region of brain
        patch_bg = patch_t < 64
        if patch_bg.sum() < 0.075 * patch_w * patch_h:
            acceptable = True

    return patch_t

def get_random_patch(nii):
    [img_h, img_w, img_d] = nii.shape
    # drop the bottom 25% and top 10% of the slices
    nii = nii[:, :, int(img_d / 4):int(9 * img_d / 10)]
    nii = _get_acceptable(nii)

    # random rotation to the patch
    # rot_angle = 45 * random.randint(0, 3)
    # nii = scipy.ndimage.rotate(nii, angle=rot_angle, reshape=True)
    # resize
    nii = scipy.misc.imresize(nii, (224, 224))
    # convert to pytorch tensor
    nii = torch.tensor(nii)
    nii.unsqueeze_(0)
    nii = nii.repeat(3, 1, 1)
    # return the mri patch and associated label
    return nii

def compute_perf(phase='test', batch_size=32):
    if phase=='train':
        data_list_path = train_csv
    elif phase=='val':
        data_list_path = val_csv
    else:
        data_list_path = test_csv

    data_list_df = pd.read_csv(data_list_path, header=None)
    data_list_df.columns = ['path', 'label']
    image_path_list = list(data_list_df['path'])
    image_label_list = list(data_list_df['label'])
    n_subjects = len(image_path_list)

    # >>>> Model >>>>
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    checkpoint = torch.load(ckpt_path)
    model_ft.load_state_dict(checkpoint['model_state_dict'])

    model_ft = model_ft.to(device)
    model_ft.eval()
    # <<<<<<<<<<<<<<<
    # image_label_list = torch(image_label_list).to(device, dtype=torch.long)
    pred_label_list = []
    for idx, path in enumerate(image_path_list):
        label = image_label_list[idx]
        nii = nibabel.load(image_path_list[idx])
        nii = nii.get_fdata()
        [img_h, img_w, img_d] = nii.shape
        # drop the bottom 25% and top 10% of the slices
        nii = nii[:, :, int(img_d / 4):int(9 * img_d / 10)]
        batch = []
        for itr in range(batch_size):
            batch.append(get_random_patch(nii))
        data = torch.stack(batch)
        data = data.to(device, dtype=torch.float)

        output = model_ft(data)
        _, preds = torch.max(output, 1)
        # print(f'label: {label}')
        # print(preds)
        preds = preds.cpu().numpy()
        preds = preds.tolist()
        pred_label = max(set(preds), key=preds.count)
        # print(pred_label)
        pred_label_list.append(pred_label)

    image_label_list = torch.tensor(image_label_list)
    pred_label_list = torch.tensor(pred_label_list)
    # print(image_label_list)
    # print(pred_label_list)
    n_correct = torch.sum(image_label_list==pred_label_list)
    n_correct = n_correct.cpu().numpy()
    acc = float(n_correct)/n_subjects
    # print(f'Accuracy: {acc}')
    return acc


if __name__=='__main__':
    print(ckpt_path)
    for phase in ['train', 'val', 'test']:
        acc = compute_perf(phase)
        print(f'{phase}: {acc}')



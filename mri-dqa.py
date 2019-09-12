"""
mri-dqa
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
import nibabel

train_csv = 'train.csv'
val_csv = 'val.csv'
n_epoch = 10000
patch_h = 56
patch_w = 56
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -- DataSet Class -----------------------------------
class MRIData(Dataset):
    def __init__(self, phase='train'):
        self.phase = phase
        if self.phase=='train':
            self.data_list_path = train_csv
        elif self.phase=='val':
            self.data_list_path = val_csv
        else:
            assert False, 'Invalid argument for phase. Choose from (train, val)'
        
        data_list_df = pd.read_csv(self.data_list_path, header=None)
        data_list_df.columns = ['path', 'label']
        self.image_path_list = list(data_list_df['path'])
        self.image_label_list = list(data_list_df['label'])
    
    def __getitem__(self, index):
        """
        Returns a patch of a slice from MRI volume
        The volume is selected by the inpurt argument index. The slice is randomly selected.
        The cropped patch is randomly selected.
        """
        nii = nibabel.load(self.image_path_list[index])
        label = self.image_label_list[index]
        nii = nii.get_fdata()
        [img_h, img_w, img_d] = nii.shape
        # drop the bottom 25% and top 10% of the slices
        nii = nii[:,:,int(img_d/4):int(9*img_d/10)]
        [img_h, img_w, img_d] = nii.shape
        # extract random slice and random patch
        h_l = int(random.randint(0, img_h-patch_h))
        h_u = int(h_l+patch_h-1)
        w_l = int(random.randint(0, img_w-patch_w))
        w_u = int(w_l+patch_w-1)
        d = int(random.randint(0, img_d-1))
        nii = nii[h_l:h_u, w_l:w_u, d]
        # resize
        nii = scipy.misc.imresize(nii, (224, 224))
        # convert to pytorch tensor
        nii = torch.tensor(nii)
        nii.unsqueeze_(0)
        nii = nii.repeat(3,1,1)
        # return the mri patch and associated label
        return nii, label

    def __len__(self):
        return len(self.image_label_list)

# -- DataSet Class -----------------------------------

def train_model(model, criterion, optimizer, scheduler, num_epochs=n_epoch):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode

                dataset = MRIData(phase)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=True)
            else:
                model.eval()   # Set model to evaluate mode
                dataset = MRIData(phase)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=True)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for ibatch, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (ibatch*32)
            epoch_acc = running_corrects.double() / (ibatch*32)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)


if __name__=='__main__':
    main()

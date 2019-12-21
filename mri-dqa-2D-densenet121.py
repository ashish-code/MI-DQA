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
import scipy.ndimage
import time 
import copy

# Nifti I/O
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    import nibabel


train_csv = 'train-office.csv'
val_csv = 'val-office.csv'
n_epoch = 250
patch_h = 56
patch_w = 56

checkpoint_dir = './checkpoints/'
ckpt_path = checkpoint_dir+'mri-dqa-2d-densenet121.pth'
perf_path = checkpoint_dir+'mri-dqa-2d-densenet121.perf'
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

    def _get_acceptable(self, patch):
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
        nii = self._get_acceptable(nii)

        # random rotation to the patch
        rot_angle = 45*random.randint(0, 3)
        nii = scipy.ndimage.rotate(nii, angle=rot_angle, reshape=True)
        # resize
        nii = scipy.misc.imresize(nii, (224, 224))
        # convert to pytorch tensor
        nii = torch.tensor(nii)
        nii.unsqueeze_(0)
        nii = nii.repeat(3, 1, 1)
        # return the mri patch and associated label
        return nii, label

    def __len__(self):
        return len(self.image_label_list)

# -- DataSet Class -----------------------------------

def train_model(model, criterion, optimizer, scheduler, epoch, perf, num_epochs=n_epoch):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    while epoch < num_epochs:
        epoch += 1
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase=='train':
                perf['train_loss'].append(epoch_loss)
                perf['train_acc'].append(epoch_acc)
            else:
                perf['val_loss'].append(epoch_loss)
                perf['val_acc'].append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()

        # save performance
        torch.save({'train_loss': perf['train_loss'],
                    'train_acc': perf['train_acc'], 'val_loss': perf['val_loss'],
                    'val_acc': perf['val_acc']}, perf_path)

        # save checkpoint every 10 epochs
        if epoch%10 == 0:
            print(' -- writing checkpoint and performance files -- ')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(), 'loss': loss,
            'scheduler': scheduler.state_dict()}, ckpt_path)



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    # If pretrained model is saved, use it
    if not os.path.exists(ckpt_path):
        model_ft = models.densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)
        epoch = 0
        perf = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    else:
        model_ft = models.densenet121(pretrained=False)
        num_ftrs = model_ft.classifier.in_features
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


if __name__ == '__main__':
    main()

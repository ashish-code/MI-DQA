"""
MRI-DQA-3D-ResNet-18

Image Quality Assessment for MRI using 3D Tensors and ResNet-18 model

author: ashish gupta
email: ashishagupta@gmail.com
"""

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
from scipy.ndimage import zoom
import time 
import copy
from functools import partial

# Nifti I/O
import nibabel

# supress dicom verison warning
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

train_csv = 'train-office.csv'
val_csv = 'val-office.csv'
n_epoch = 100000
patch_h = 56
patch_w = 56
patch_d = 16
batch_size = 24

checkpoint_dir = './checkpoints/'
pretrained_dir = './pretrained/'
ckpt_path = checkpoint_dir+'mri-dqa-3d-resnet-18.pth'
perf_path = checkpoint_dir+'mri-dqa-3d-resnet-18.perf'
pretrained_path = pretrained_dir+'resnet-18-kinetics.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

__all__ = ['ResNet', 'resnet18']

# >> Model Class >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='A', num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3,64, kernel_size=7,stride=(1, 2, 2),padding=(3, 3, 3),bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

# << Model Class <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



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
        h_u = int(h_l+patch_h)
        w_l = int(random.randint(0, img_w-patch_w))
        w_u = int(w_l+patch_w)
        d_l = int(random.randint(0, img_d-patch_d))
        d_u = int(d_l+patch_d)
        nii = nii[h_l:h_u, w_l:w_u, d_l:d_u]
        # resize
        # nii = scipy.misc.imresize(nii, (224, 224))
        nii = zoom(nii, (2,2,1))
        # convert to pytorch tensor
        nii = torch.tensor(nii)
        nii.unsqueeze_(0)
        nii = nii.repeat(3,1,1,1)
        # permute the data to put 
        nii = nii.permute(0, 3, 1, 2)
        # return the mri patch and associated label
        return nii, label

    def __len__(self):
        return len(self.image_label_list)

# -- DataSet Class -----------------------------------

def train_model(model, criterion, optimizer, scheduler, epoch, perf, num_epochs):
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
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
            else:
                model.eval()   # Set model to evaluate mode
                dataset = MRIData(phase)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

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

            epoch_loss = running_loss / (ibatch*batch_size)
            epoch_acc = running_corrects.double() / (ibatch*batch_size)

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

        # save checkpoint
        if epoch%2 == 0:
            print(' -- writing checkpoint and performance files -- ')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(), 
            'scheduler': scheduler.state_dict()}, ckpt_path)

            torch.save({'train_loss': perf['train_loss'],
            'train_acc': perf['train_acc'], 'val_loss': perf['val_loss'],
            'val_acc': perf['val_acc']}, perf_path)

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    dataset = MRIData('train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    for data, label in dataloader:
        print(data.shape)
        break
    
    # instantiate model, load if pre-trained model is available.
    model = resnet18(num_classes = 400, shortcut_type='A', sample_size=112, sample_duration=16)
    

    model = model.cuda()
    # model = nn.DataParallel(model)
    # check to see if pre-trained file should be loaded
    if not os.path.isfile(ckpt_path):
        # load the pretrained
        pretrain = torch.load(pretrained_path)
        psd = pretrain['state_dict']
        psd2 = dict()
        for k,v in psd.items():
            k2 = k.replace('module.','')
            psd2[k2] = v

        model.load_state_dict(psd2)
        num_filters = model.fc.in_features
        model.fc2 = nn.Linear(num_filters, 2)
        model.fc2 = model.fc2.cuda()
        parameters = get_fine_tuning_parameters(model, 0)
        optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        epoch = 0
        loss = 0.0
        perf = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    else:
        # load the checkpoint
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        num_filters = model.fc.in_features
        model.fc2 = nn.Linear(num_filters, 2)
        model.fc2 = model.fc2.cuda()
        parameters = get_fine_tuning_parameters(model, 0)
        optimizer = optim.SGD(parameters, lr=0.0001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # load
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        exp_lr_scheduler.load_state_dict(checkpoint['scheduler'])
        perf = torch.load(perf_path)
        for state in optimizer.state.values():
            for k,v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, epoch, perf, n_epoch)


if __name__=='__main__':
    main()
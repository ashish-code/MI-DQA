"""
MRI-DQA-ResNet

Deep Quality Assessment of MRI using ResNet Model

author: Ashish Gupta
email: ashishagupta@gmail.com
"""

# DL Library
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms


import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import math

# Nifti I/O
import nibabel

train_csv = 'train.csv'
val_csv = 'val.csv'
n_epoch = 10000
patch_h = 64
patch_w = 64

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
        d = int(random.randint(0, img_d))
        nii = nii[h_l:h_u, w_l:w_u, d]
        # convert to pytorch tensor
        nii = torch.tensor(nii)
        nii = nii.view(1, patch_h, patch_w).expand(3, -1, -1)
        # return the mri patch and associated label
        return nii, label

    def __len__(self):
        return len(self.image_label_list)

# -- DataSet Class -----------------------------------

# -- Network Class -----------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

    def __init__(self, block, layers, inp_channels= 3, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, 64, kernel_size=2, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

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
# -- Network Class -----------------------------------------------

def main():
    # instantiate network object
    model  = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=2,inp_channels=3)

    print(model)

    # toggle gpu use based on availability
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        print ('USE GPU')
    else:
        print ('USE CPU')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.3, momentum = 0.1)

    train_dataset = MRIData(phase='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1, drop_last=True)

    # put model in training mode
    model.train()

    for epoch in range(n_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.to('cuda:0')
            print(data.shape)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        if epoch%1000 == 0:
            model_save_path = './checkpoints/epoch_{}.pth'.format(epoch)
            torch.save({'epoch': epoch, 'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict()},model_save_path)

    print('finished training')

# # Testing Code
if __name__=='__main__':
    main()
    
    
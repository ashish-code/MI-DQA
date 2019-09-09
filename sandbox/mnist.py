import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
net  = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10,inp_channels=1)

print(net)

use_gpu = torch.cuda.is_available()
if use_gpu:
	net = net.cuda()
	print ('USE GPU')
else:
	print ('USE CPU')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.3, momentum = 0.1)

print ("1. Loading data")
train = pd.read_csv("train.csv").values
test  = pd.read_csv("test.csv").values

print ("2. Converting data")
X_data  = train[:, 1:].reshape(train.shape[0], 1, 28, 28)
X_data  = X_data.astype(float)
X_data /= 255.0
X_data  = torch.from_numpy(X_data);
X_label = train[:,0];
X_label = X_label.astype(int);
X_label = torch.from_numpy(X_label);
X_label = X_label.view(train.shape[0],-1);
print (X_data.size(), X_label.size())

print ("3. Training phase")
nb_train = train.shape[0]
nb_epoch = 20000
nb_index = 0
nb_batch = 250

for epoch in range(nb_epoch):
	if nb_index + nb_batch >= nb_train:
		nb_index = 0
	else:
		nb_index = nb_index + nb_batch

	mini_data  = Variable(X_data[nb_index:(nb_index+nb_batch)].clone())
	mini_label = Variable(X_label[nb_index:(nb_index+nb_batch)].clone(), requires_grad = False)
	mini_data  = mini_data.type(torch.FloatTensor)
	mini_label = mini_label.type(torch.LongTensor)
	if use_gpu:
		mini_data  = mini_data.cuda()
		mini_label = mini_label.cuda()
	optimizer.zero_grad()
	mini_out   = net(mini_data)
	mini_label = mini_label.view(nb_batch)
	mini_loss  = criterion(mini_out, mini_label)
	mini_loss.backward()
	optimizer.step() 

	if (epoch + 1) % 100 == 0:
		print("Epoch = %d, Loss = %f" %(epoch+1, mini_loss.item()))

print ("4. Testing phase")

Y_data  = test.reshape(test.shape[0], 1, 28, 28)
Y_data  = Y_data.astype(float)
Y_data /= 255.0
Y_data  = torch.from_numpy(Y_data);
print (Y_data.size())
nb_test = test.shape[0]

net.eval()

final_prediction = np.ndarray(shape = (nb_test, 2), dtype=int)
for each_sample in range(nb_test):
	sample_data = Variable(Y_data[each_sample:each_sample+1].clone())
	sample_data = sample_data.type(torch.FloatTensor)
	if use_gpu:
		sample_data = sample_data.cuda()
	sample_out = net(sample_data)
	pred = torch.max(sample_out, 1)
	final_prediction[each_sample][0] = 1 + each_sample
	final_prediction[each_sample][1] = pred[1][0]
	if (each_sample + 1) % 2000 == 0:
		print("Total tested = %d" %(each_sample + 1))

print ('5. Generating submission file')

submission = pd.DataFrame(final_prediction, dtype=int, columns=['ImageId', 'Label'])
submission.to_csv('submission.csv', index=False, header=True)
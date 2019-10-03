"""
experimenting with accessing specific list of layers in the network
"""

import torch
from torch import nn
from torchvision import models
from torchvision import transforms
# from torchsummary import summary

model = models.resnet18(pretrained=True)
# print(model)

# my_model = nn.Sequential(*list(model.children())[:-1])
# print(my_model)

for i,child in enumerate(model.children()):
    print(i, torch.Size(child))
    print('--------')

# print(summary(model, (3,224,224)))

# vgg = models.vgg16()
# print(summary(vgg, (3, 224, 224)))
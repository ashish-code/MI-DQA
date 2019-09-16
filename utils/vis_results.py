"""
visualize training results

author: ashish gupta
email: ashishagupta@gmail.com
"""

import torch
import matplotlib.pyplot as plt


def vis_2d():

    file_path = 'C:/Users/Ashish/Repos/MI-DQA/checkpoints/mri-dqa-2d-resnet-18.perf'

    perf = torch.load(file_path)

    print(perf.keys())

    train_loss = perf['train_loss']
    train_acc = perf['train_acc']
    val_loss = perf['val_loss']
    val_acc = perf['val_acc']

    fig = plt.figure(1, figsize=(15,5))
    plt.subplot(2,1,1)
    plt.plot(train_loss, 'k-', label='train loss')
    plt.plot(val_loss, 'c-', label='valid loss')
    plt.xlabel('epochs')
    plt.ylabel('x-entropy loss')
    plt.title('MRI-DQA-2D-ResNet-18-ABIDE-1')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(val_acc, 'c-',label='valid accuracy')
    plt.plot(train_acc, 'k-', label='train accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def vis_3d():
    # file_path = 'C:/Users/Ashish/Repos/MI-DQA/checkpoints/mri-dqa-2d-resnet-18.perf'
    file_path = 'D:/Repos/MI-DQA/checkpoints/mri-dqa-3d-resnet-18.perf'
    perf = torch.load(file_path)
    train_loss = perf['train_loss']
    train_acc = perf['train_acc']
    val_loss = perf['val_loss']
    val_acc = perf['val_acc']

    fig = plt.figure(1, figsize=(15,5))
    plt.subplot(2,1,1)
    plt.plot(train_loss, 'k-', label='train loss')
    plt.plot(val_loss, 'c-', label='valid loss')
    plt.xlabel('epochs')
    plt.ylabel('x-entropy loss')
    plt.title('MRI-DQA-3D-ResNet-18-ABIDE-1')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(val_acc, 'c-',label='valid accuracy')
    plt.plot(train_acc, 'k-', label='train accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


if __name__=='__main__':
    vis_3d()
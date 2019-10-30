"""
The ResNet-18 model (pre-trained) is used as a feature extractor.
The average pooling layer in the last layer before the softmax FC layer is used as a
feature vector of dims: 512.
The feature vector and associated label is used to train an isolation forest classifier.

author: ashish gupta
email: ashishagupta@gmail.com
date: 10-28-2019
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
import skimage.transform
from skimage import color
from skimage import io
from skimage import img_as_float
import cv2

#Isolation Forest
from sklearn.ensemble import IsolationForest

# Nifti I/O
import nibabel

train_csv = 'train-office.csv'
val_csv = 'val-office.csv'
n_epoch = 500
patch_h = 56
patch_w = 56

checkpoint_dir = './checkpoints/'
ckpt_path = checkpoint_dir+'mri-dqa-2d-resnet-18-rot-onbrain.pth'
perf_path = checkpoint_dir+'mri-dqa-2d-resnet-18-rot-onbrain.perf'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

feature_train_csv = 'resnet18_feature_train.csv'
feature_val_csv = 'resnet18_feature_val.csv'

scaler = transforms.Scale((224, 224))


class Patch:
    def __init__(self, mri_slice, h_l, h_u, w_l, w_u):
        self.mri_slice = mri_slice
        self.h_l = h_l
        self.h_u = h_u
        self.w_l = w_l
        self.w_u = w_u


# >> DataSet Class >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class MRIData:
    def __init__(self, phase=0):
        self.phase = phase
        if self.phase == 0:
            self.data_list_path = train_csv
        elif self.phase == 1:
            self.data_list_path = val_csv
        else:
            assert False, 'Invalid argument for phase. Choose from (0, 1)'

        data_list_df = pd.read_csv(self.data_list_path, header=None)
        data_list_df.columns = ['path', 'label']
        self.image_path_list = list(data_list_df['path'])
        self.image_label_list = list(data_list_df['label'])

    def _get_acceptable(self, patch):
        [img_h, img_w, img_d] = patch.shape
        # extract random mri_slice and random patch
        acceptable = False
        while not acceptable:
            h_l = int(random.randint(0, img_h - patch_h))
            h_u = int(h_l + patch_h - 1)
            w_l = int(random.randint(0, img_w - patch_w))
            w_u = int(w_l + patch_w - 1)
            d = int(random.randint(0, img_d - 1))
            patch_t = patch[h_l:h_u, w_l:w_u, d]
            mri_slice = patch[:,:,d]
            mri_patch = Patch(mri_slice, h_l, h_u, w_l, w_u)
            # select patch if overlapping sufficient region of brain
            patch_bg = patch_t < 64
            if patch_bg.sum() < 0.075 * patch_w * patch_h:
                acceptable = True

        return patch_t, mri_patch

    def getitem(self, index):
        """
        Returns a patch of a slice from MRI volume
        The volume is selected by the input argument index. The slice is randomly selected.
        The cropped patch is randomly selected.
        """
        nii = nibabel.load(self.image_path_list[index])
        label = self.image_label_list[index]
        nii = nii.get_fdata()
        [img_h, img_w, img_d] = nii.shape
        # drop the bottom 25% and top 10% of the slices
        nii = nii[:, :, int(img_d / 4):int(9 * img_d / 10)]
        nii, mri_patch = self._get_acceptable(nii)
        # resize
        nii = skimage.transform.resize(nii, (224, 224))
        # convert to pytorch tensor
        nii = torch.tensor(nii)
        nii.unsqueeze_(0)
        nii = nii.repeat(3, 1, 1)
        # return the mri patch and associated label
        return nii, label

    def len(self):
        return len(self.image_path_list)
# << DataSet Class <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def get_vector(model, layer, img):
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable((img.unsqueeze(0)).cuda(), requires_grad=True)
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(1, 512, 1, 1)
    # 4. Define a function that will copy the output of a layer

    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding


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
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    layer = model._modules.get('avgpool')
    # print(layer)
    # put model in evaluation mode
    model.eval()
    for phase in [0, 1]:
        dataset = MRIData(phase)
        n_items = dataset.len()

        feat_mat = dict()
        for count in range(n_items):
            image, label = dataset.getitem(count)
            # print(image.shape)
            image = image.to(device, dtype=torch.float)
            embedding_v = get_vector(model, layer, image)
            feature_v = embedding_v[0, :, 0, 0]
            feature_v = feature_v.cpu().detach().numpy().tolist()
            feature_v.append(label)
            feat_mat[count] = feature_v
        df = pd.DataFrame(feat_mat)
        df = df.transpose()
        print(phase, df.shape)

        if phase == 0:
            feature_csv = feature_train_csv
        elif phase == 1:
            feature_csv = feature_val_csv
        else:
            print('phase not recognized')
        df.to_csv(feature_csv, header=False, index=False)
        print(f'written to {feature_csv}')

rng = np.random.RandomState(123)

def iso_forest():
    """
    Compute the results of the isolation forest on embedded feature vectors
    :return:
    """
    # data_train = np.genfromtxt(feature_train_csv, delimiter=',', dtype=float)
    # X_data = data_train[:, :-1]
    # y_train = data_train[:, -1]
    # data_val = np.genfromtxt(feature_val_csv, delimiter=',', dtype=float)
    # X_data = data_val[:, :-1]
    # y_val = data_val[:, -1]
    # X_train = X_data[y_train==0.0,:]
    # X_outliers = X_data[y_train==1.0,:]

    data = pd.read_csv(feature_train_csv, header=None, index_col=None)
    X_train = data[data.iloc[:,512] == 1.0].as_matrix()[:,:-1]
    X_outliers = data[data.iloc[:,512] == 0.0].as_matrix()[:,:-1]

    data = pd.read_csv(feature_val_csv, header=None, index_col=None)
    X_test = data[data.iloc[:, 512] == 1.0].as_matrix()[:, :-1]

    clf = IsolationForest(behaviour='new', max_samples=250, random_state=rng, contamination=0.5)
    clf.fit(X_train[:,256:258])

    # y_pred_train = clf.predict(X_train)
    # y_pred_test = clf.predict(X_test)
    # y_pred_outliers = clf.predict(X_outliers)

    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy = np.meshgrid(np.linspace(0, 4, 50), np.linspace(0, 4, 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("IsolationForest")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', alpha=0.6,
                     s=20, edgecolor='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green', alpha=0.6,
                     s=20, edgecolor='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', alpha=0.6,
                    s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((0, 4))
    plt.ylim((0, 4))
    plt.legend([b1, b2, c],
               ["training acceptable",
                "validation acceptable", "training artifact"],
               loc="upper right")
    plt.show()


if __name__ == '__main__':
    # main()
    iso_forest()

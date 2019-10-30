"""
Class Activation Map for ResNet-18 with 2D Tensor of MRI slices
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from torchvision import models
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn import functional as F
from torch import topk
import numpy as np

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    import nibabel

import os
import random
import pandas as pd
import math
import scipy.misc
import time
import copy
import skimage.transform

patch_h = 56
patch_w = 56
# train_csv = 'utils/train-cam-tesla-0.csv'
# val_csv = 'utils/val-cam-tesla-0.csv'
train_0_csv = 'utils/train-cam-office-0.csv'
val_0_csv = 'utils/val-cam-office-0.csv'

train_1_csv = 'utils/train-cam-office-1.csv'
val_1_csv = 'utils/val-cam-office-1.csv'

checkpoint_dir = './checkpoints/'
ckpt_path = checkpoint_dir+'mri-dqa-2d-resnet-18-rot-onbrain.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# >> DataSet Class >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
class MRIData(Dataset):
    def __init__(self, phase=0):
        self.phase = phase
        if self.phase == 0:
            self.data_list_path = train_0_csv
        elif self.phase == 1:
            self.data_list_path = train_1_csv
        else:
            assert False, 'Invalid argument for phase. Choose from (0, 1)'

        data_list_df = pd.read_csv(self.data_list_path, header=None)
        data_list_df.columns = ['path']
        self.image_path_list = list(data_list_df['path'])

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
            slice = patch[:,:,d]
            # select patch if overlapping sufficient region of brain
            patch_bg = patch_t < 64
            if patch_bg.sum() < 0.075 * patch_w * patch_h:
                acceptable = True

        return patch_t, slice

    def __getitem__(self, index):
        """
        Returns a patch of a slice from MRI volume
        The volume is selected by the input argument index. The slice is randomly selected.
        The cropped patch is randomly selected.
        """
        nii = nibabel.load(self.image_path_list[index])
        nii = nii.get_fdata()
        [img_h, img_w, img_d] = nii.shape
        # drop the bottom 25% and top 10% of the slices
        nii = nii[:, :, int(img_d / 4):int(9 * img_d / 10)]
        nii, slice = self._get_acceptable(nii)
        # resize
        nii = scipy.misc.imresize(nii, (224, 224))
        # convert to pytorch tensor
        nii = torch.tensor(nii)
        nii.unsqueeze_(0)
        nii = nii.repeat(3, 1, 1)
        # return the mri patch and associated label
        return nii, slice

    def __len__(self):
        return len(self.image_path_list)

# << DataSet Class <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class SaveFeatures():
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()


def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h,w)
    cam = cam - np.min(cam)
    cam_img = cam/np.max(cam)
    return [cam_img]


def grad_cam(image, model, count):
    prediction_var = Variable((image.unsqueeze(0)).cuda(), requires_grad=True)
    # reference to the final layers, depends on the model class
    final_layer = model._modules.get('layer4')
    activated_features = SaveFeatures(final_layer)
    # put the flattened input image through the model
    prediction = model(prediction_var)
    pred_probabilities = F.softmax(prediction).data.squeeze()
    activated_features.remove()
    topk_pred = topk(pred_probabilities, 1)
    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    class_idx = topk(pred_probabilities, 1)[1].int()
    overlay = getCAM(activated_features.features, weight_softmax, class_idx)

    img = image[0, :, :].cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img, cmap=plt.cm.bone)
    ax[0].set_xticks([], [])
    ax[0].set_yticks([], [])
    ax[0].set_title('MRI Slice Patch')
    ax[1].imshow(img, cmap=plt.cm.gray)
    ax[1].imshow(skimage.transform.resize(overlay[0], image.shape[1:3]), alpha=0.3, cmap='jet')
    ax[1].set_xticks([], [])
    ax[1].set_yticks([], [])
    ax[1].set_title('Grad-CAM MRI')
    fig.suptitle('Grad-CAM MRI-DQA-Augmented ResNet-18-ABIDE-1')
    fig_path = f'gradcam_rot_onbrain/gradcam-{count+1}.png'
    plt.savefig(fig_path)
    print(fig_path)

def main():
    # Get the pretrained model from checkpoint
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    # put model in evaluation mode
    model.eval()
    phase = 0
    dataset = MRIData(phase)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    for count, (inputs, slice) in enumerate(dataloader):
        image = inputs[0,:,:,:]
        image = image.to(device, dtype=torch.float)
        grad_cam(image, model, count)


if __name__=='__main__':
    main()


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
from skimage import color
from skimage import io
from skimage import img_as_float
import cv2
import seaborn as sns

patch_h = 56
patch_w = 56
train_csv = 'train-office.csv'
val_csv = 'val-office.csv'
# train_csv = 'utils/train-cam-tesla-0.csv'
# val_csv = 'utils/val-cam-tesla-0.csv'
train_0_csv = 'utils/train-cam-office-0.csv'
val_0_csv = 'utils/val-cam-office-0.csv'

train_1_csv = 'utils/train-cam-office-1.csv'
val_1_csv = 'utils/val-cam-office-1.csv'

checkpoint_dir = './checkpoints/'
ckpt_path = checkpoint_dir+'mri-dqa-2d-resnet-18-rot-onbrain.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CAM_MIN = -5.0
CAM_MAX = 30.0


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
        # random shuffle the rows
        data_list_df.sample(frac=1).reset_index(drop=True)
        self.image_path_list = list(data_list_df['path'])
        self.image_label_list = list(data_list_df['label'])

    def _get_acceptable(self, volume):
        [img_h, img_w, img_d] = volume.shape
        # extract random mri_slice and random patch
        acceptable = False
        while not acceptable:
            h_l = int(random.randint(0, img_h - patch_h))
            h_u = int(h_l + patch_h - 1)
            w_l = int(random.randint(0, img_w - patch_w))
            w_u = int(w_l + patch_w - 1)
            d = int(random.randint(0, img_d - 1))
            _slice = volume[:, :, d]
            mri_slice = volume[h_l:h_u, w_l:w_u, d]
            mri_patch = Patch(mri_slice, h_l, h_u, w_l, w_u)
            # select patch if overlapping sufficient region of brain
            patch_bg = mri_slice < 64
            if patch_bg.sum() < 0.075 * patch_w * patch_h:
                acceptable = True

        return _slice, mri_patch

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
        _slice, mri_patch = self._get_acceptable(nii)
        # resize
        nii = skimage.transform.resize(mri_patch.mri_slice, (224, 224))
        # convert to pytorch tensor
        nii = torch.tensor(nii)
        nii.unsqueeze_(0)
        nii = nii.repeat(3, 1, 1)
        # return the mri patch and associated label
        return nii, mri_patch, _slice, label

    def len(self):
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
    cam_img = cam
    # cam_img = cam_img - np.min(cam_img)
    cam_img = cam_img - CAM_MIN
    # cam_img = cam_img/np.max(cam)
    cam_img = cam_img/CAM_MAX
    cam_img = cam_img.reshape((h, w))
    return cam_img, cam


def grad_cam(image, _slice, mri_patch, model, _count, label):
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
    overlay, cam_raw = getCAM(activated_features.features, weight_softmax, class_idx)

    # print(f'cam shape: {overlay.shape}')
    # img = image[0, :, :].cpu().numpy()
    img = mri_patch.mri_slice
    # print(f'mri_slice shape: {img.shape}')
    img_h, img_w = img.shape
    # print(f'type img: {type(img)}')
    h_l = mri_patch.h_l
    h_u = mri_patch.h_u
    w_l = mri_patch.w_l
    w_u = mri_patch.w_u
    # overlay_resized = skimage.transform.resize(overlay, (patch_h, patch_w))
    overlay_resized = cv2.resize(overlay, (patch_h, patch_w))
    overlay_resized = np.uint8(255 * (overlay_resized - np.min(overlay_resized)) / np.max(overlay_resized))
    overlay_resized = cv2.applyColorMap(overlay_resized, cv2.COLORMAP_JET)
    my_slice = np.uint8(255*((_slice-np.min(_slice))/np.max(_slice)))
    # my_slice = cv2.applyColorMap(my_slice, cv2.COLORMAP_BONE)
    my_slice = cv2.cvtColor(my_slice, cv2.COLOR_GRAY2RGB)

    # img = np.uint8(255*img)
    # img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    # plt.imshow(overlay_resized)
    # plt.show()
    # img_2 = img.copy()
    # img_2 = img_2[h_l:h_u+1, w_l:w_u+1]
    # cv2.addWeighted(img[h_l:h_u+1, w_l:w_u+1], 1.0, overlay_resized, 0.3, 0.0, img_2)
    temp = 0.5*overlay_resized+ 0.5*my_slice[h_l:h_u + 1, w_l:w_u + 1]
    # temp = temp/np.max(temp)
    # temp = np.uint8(255*temp)
    # img[h_l:h_u + 1, w_l:w_u + 1] = temp
    # plt.show()
    # print(f'slice shape: {my_slice.shape}')
    # print(f'overlay shape: {overlay_resized.shape}')
    my_slice[h_l:h_u + 1, w_l:w_u + 1] = temp
    # fig = plt.figure()
    # cv2.imshow('grad-cam ', my_slice)
    # cv2.waitKey(30)
    fig_path = f'gradcam_normed/gradcam-{_count}-{label}.png'
    # plt.savefig(fig_path)
    # print(label, fig_path)
    # cv2.imwrite(fig_path, my_slice)
    return cam_raw


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
    n_items = dataset.len()
    # debug:
    # n_items = 1
    cam_val = np.zeros((n_items, 50))
    for count in range(n_items):
        image, mri_patch, _slice, label = dataset.getitem(count)
        image = image.to(device, dtype=torch.float)
        cam_raw = grad_cam(image, _slice, mri_patch, model, count, label)

        # print(f'label: {label}, max: {np.max(cam_raw)}, min:{np.min(cam_raw)}')
        # cam_val.append(cam_raw.tolist())
        cam_val[count, :-1] = cam_raw
        cam_val[count, -1] = label
    cam_df = pd.DataFrame(cam_val)
    cam_df.to_csv('sandbox/cam_df.csv')

    # print('\n-------------------------------\n')
    # print(np.max(cam_val))
    # print(np.min(cam_val))


if __name__ == '__main__':
    main()


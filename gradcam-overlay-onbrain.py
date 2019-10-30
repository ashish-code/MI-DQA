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
            self.data_list_path = train_0_csv
        elif self.phase == 1:
            self.data_list_path = train_1_csv
        else:
            assert False, 'Invalid argument for phase. Choose from (0, 1)'

        data_list_df = pd.read_csv(self.data_list_path, header=None)
        data_list_df.columns = ['path']
        self.image_path_list = list(data_list_df['path'])

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
        return nii, mri_patch, _slice

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
    cam_img = cam_img - np.min(cam_img)
    cam_img = cam/np.max(cam)
    cam_img = cam_img.reshape((h,w))
    return [cam_img]


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


def grad_cam(image, _slice, mri_patch, model, _count):
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
    overlay = getCAM(activated_features.features, weight_softmax, class_idx)[0]
    # plt.imshow(overlay)
    # plt.show()
    print(f'cam shape: {overlay.shape}')
    # img = image[0, :, :].cpu().numpy()
    img = mri_patch.mri_slice
    print(f'mri_slice shape: {img.shape}')
    img_h, img_w = img.shape
    print(f'type img: {type(img)}')
    h_l = mri_patch.h_l
    h_u = mri_patch.h_u
    w_l = mri_patch.w_l
    w_u = mri_patch.w_u
    # overlay_resized = skimage.transform.resize(overlay, (patch_h, patch_w))
    overlay_resized = cv2.resize(overlay, (patch_h, patch_w))
    overlay_resized = np.uint8(255*overlay_resized)
    overlay_resized = cv2.applyColorMap(overlay_resized, cv2.COLORMAP_JET)
    my_slice = np.uint8(255*_slice)
    # my_slice = cv2.applyColorMap(my_slice, cv2.COLORMAP_BONE)
    my_slice = cv2.cvtColor(my_slice, cv2.COLOR_GRAY2RGB)

    img = np.uint8(255*img)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
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
    print(f'slice shape: {my_slice.shape}')
    print(f'overlay shape: {overlay_resized.shape}')
    my_slice[h_l:h_u + 1, w_l:w_u + 1] = temp
    fig = plt.figure()
    cv2.imshow('grad-cam ', my_slice)
    cv2.waitKey(30)

    # plt.xticks([], [])
    # plt.yticks([], [])
    # plt.title('Gradient Class Activation Map')
    fig_path = f'gradcam_onbrain_overlay/gradcam-{_count}.png'
    # plt.savefig(fig_path)
    print(fig_path)
    cv2.imwrite(fig_path, my_slice)

    # background = background.cpu().numpy()

    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # ax[0].imshow(img, cmap=plt.cm.bone)
    # ax[0].set_xticks([], [])
    # ax[0].set_yticks([], [])
    # ax[0].set_title('MRI Slice Patch')
    # ax[1].imshow(img, cmap=plt.cm.gray)
    # ax[1].imshow(skimage.transform.resize(overlay, image.shape[1:3]), alpha=0.3, cmap='jet')
    # ax[1].set_xticks([], [])
    # ax[1].set_yticks([], [])
    # ax[1].set_title('Grad-CAM MRI')
    # fig.suptitle('Grad-CAM MRI-DQA-Augmented ResNet-18-ABIDE-1')
    # fig_path = f'gradcam_rot_onbrain/gradcam-{count+1}.png'
    # plt.savefig(fig_path)
    # print(fig_path)


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
    for count in range(n_items):
        image, mri_patch, _slice = dataset.getitem(count)
        image = image.to(device, dtype=torch.float)
        grad_cam(image, _slice, mri_patch, model, count)

        # if count >= 0:
        #     break

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    # for count, (inputs, mri_patch) in enumerate(dataloader):
    #     image = inputs[0,:,:,:]
    #     image = image.to(device, dtype=torch.float)
    #     grad_cam(image, mri_patch, model, count)
    #     if count > 1:
    #         break


if __name__=='__main__':
    main()


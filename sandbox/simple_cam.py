"""
toy problem CAM with pytorch
"""

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform

image = Image.open('cat2.png')

# transform image to required format for ResNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
display_transform = transforms.Compose([transforms.Resize((224, 224))])

# apply transformations to input image
tensor = preprocess(image)
prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)

# ResNet18 pretrained model
model = models.resnet18(pretrained=True)
# put model onto cuda
model.cuda()
# put model in evaluation mode
model.eval()
# activation features are deleted so a hook is required to save them
class SaveFeatures():
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self):
        self.hook.remove()

# reference to the final layers, depends on the model class
final_layer = model._modules.get('layer4')
activated_features = SaveFeatures(final_layer)

# put the flattened input image through the model
prediction = model(prediction_var)
pred_probabilities = F.softmax(prediction).data.squeeze()
activated_features.remove()

topk_pred = topk(pred_probabilities, 1)

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h,w)
    cam = cam - np.min(cam)
    cam_img = cam/np.max(cam)
    return [cam_img]


weight_softmax_params = list(model._modules.get('fc').parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

# print(weight_softmax_params)

class_idx = topk(pred_probabilities,1)[1].int()

overlay = getCAM(activated_features.features, weight_softmax, class_idx)

plt.figure(1)
plt.imshow(overlay[0], alpha=0.5, cmap='jet')
plt.show()

plt.figure(2)
plt.imshow(display_transform(image))
plt.imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet')
plt.show()


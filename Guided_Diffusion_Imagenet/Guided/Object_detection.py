from pycocotools.coco import COCO
import numpy as np
import random
import os
import cv2

import sys
sys.path.append('./')

### For visualizing the outputs ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms, utils
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights, deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import random
from typing import Any, Callable, List, Optional, Tuple
import torch.optim as optim
from torch import nn, einsum
from torch.autograd import Variable

import torch
import torch.utils.data

from Guided.dataset.helpers import get_splitted_dataset
from Guided.helpers import get_model, get_parser, Operation, OptimizerDetails
from Guided.helpers import Jitter, ColorJitterR
from Guided.models.resnet import ResNet18_64x64, ResNet18_64x64_1, ResNet18_256x256
from scripts.imagenet import get_loader_from_dataset, get_train_val_datasets
import cv2


import torchvision
import cv2
from torchvision import transforms, utils
from torch.utils import data
import torch.nn.functional as F
import torchvision.models as models
import os
import errno
import shutil



def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

torch.manual_seed(0)

parser = get_parser()
parser.add_argument('--root', default='/fs/cml-datasets/ImageNet/ILSVRC2012')
parser.add_argument("--partitions_folder", default="/cmlscratch/bansal01/fall_2022/Guided_Diffusion_Imagenet/checkpoints/non_equal_split/")
#parser.add_argument("--partitions_folder", default="checkpoints/non_equal_split/")
parser.add_argument('--optimizer', default='adamw', choices=['sgd', 'adamw'])
parser.add_argument("--optim_lr", default=1e-3, type=float)
parser.add_argument('--optim_backward_guidance_max_iters', type=int, default=1)
parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
parser.add_argument('--optim_original_conditioning', action='store_true', default=False)
parser.add_argument("--optim_forward_guidance_wt", default=2.0, type=float)
parser.add_argument('--optim_warm_start', action='store_true', default=False)
parser.add_argument('--optim_print', action='store_true', default=False)
parser.add_argument('--optim_aug', action='store_true', default=False)
parser.add_argument('--optim_folder', default='/cmlscratch/hmchu/guided-diffusion/')
parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
parser.add_argument("--exp_name", default='')
parser.add_argument("--box_from_sample_image", action='store_true')
parser.add_argument("--indexes", nargs="+", default=[2, 9, 16, 24, 68, 164, 238], type=int)
parser.add_argument("--trials", default=10, type=int)
parser.add_argument("--n_samples", default=10, type=int)

args = parser.parse_args()

BATCH_SIZE = args.batch_size
assert BATCH_SIZE == 1
resolution_fact = 8

from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

class ObjectDetection(nn.Module):
    def __init__(self):
        super().__init__()
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
        self.preprocess = weights.transforms()
        for param in self.model.parameters():
            param.requires_grad = False
        self.categories = weights.meta["categories"]

        print(weights.meta["categories"])


    def forward(self, x):
        self.model.eval()
        inter = self.preprocess((x+1) * 0.5)
        return self.model(inter)
    
    def cal_loss(self, x, gt):
        def set_bn_to_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()


        self.model.train()
        self.model.backbone.eval()
        self.model.apply(set_bn_to_eval)
        inter = self.preprocess((x+1) * 0.5)
        loss = self.model(inter, gt)
        return loss['loss_classifier'] + loss['loss_objectness'] + loss['loss_rpn_box_reg']
     


operation_func = ObjectDetection()
operation_func = torch.nn.DataParallel(operation_func).cuda()
operation_func.eval()

for param in operation_func.parameters():
    param.requires_grad = False

results_folder = os.path.join(args.optim_folder, args.exp_name)
create_folder(results_folder)

operation = OptimizerDetails()


operation.num_steps = args.optim_num_steps #[2]
operation.operation_func = operation_func
operation.optimizer = 'Adam'
operation.lr = args.optim_lr #0.01
operation.loss_func = None
operation.max_iters = args.optim_backward_guidance_max_iters #00
operation.loss_cutoff = args.optim_loss_cutoff #0.00001
operation.tv_loss = None
operation.guidance_3 = args.optim_forward_guidance #True
operation.original_guidance = args.optim_original_conditioning
operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
operation.warm_start = args.optim_warm_start #False
operation.print = args.optim_print
operation.print_every = 5
operation.folder = results_folder
operation.Aug = None


operator = Operation(args, operation=operation, shape=[BATCH_SIZE, 3, 256, 256], progressive=True)
cnt = 0

def return_cv2(img, path):
    black = [255, 255, 255]
    img = (img + 1) * 0.5
    utils.save_image(img, path, nrow=1)
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
    return img

print('loading the dataset...')
print('done')
print('splitting the dataset...')

train_dataset, val_dataset = get_train_val_datasets(args)
train_ckpt_path = os.path.join(args.partitions_folder, "partitions_train.pt")
val_ckpt_path = os.path.join(args.partitions_folder, "partitions_val.pt")

train1, train2 = get_splitted_dataset(dataset=train_dataset, checkpoint_path=train_ckpt_path)
val1, val2 = get_splitted_dataset(dataset=val_dataset, checkpoint_path=val_ckpt_path)
print('done')
train1, train2 = get_loader_from_dataset(args, train1, True), get_loader_from_dataset(args, train2, False)
val1, val2 = get_loader_from_dataset(args, val1, True), get_loader_from_dataset(args, val2, False)


obj_categories = operation_func.module.categories

def draw_box(img, pred):
    labels = [obj_categories[j] for j in pred["labels"].cpu()]
    uint8_image = ((img.cpu() + 1.0) / 2 * 255).to(torch.uint8)
    box = draw_bounding_boxes(uint8_image, boxes=pred["boxes"].cpu(),
            labels=labels,
            colors="red",
            width=4, font_size=30)
    box = box.float() / 255.0
    return box

print(f"indices of images to compute groun truth based on: {args.indexes}")
for batch_ind, batch in enumerate(val1):

    if batch_ind == args.n_samples - 1:
        break

    image, label = batch
    image, label = image.cuda(), label.cuda()
    
    if batch_ind not in args.indexes:
        continue

    print(f'current image index: {batch_ind}')
    
    with torch.no_grad():
        gt_boxes = operation_func(image)

    box_image = draw_box(image[0], gt_boxes[0])
    utils.save_image(box_image, f'{results_folder}/box_og_img_{batch_ind}.png')

    for rep in range(args.trials):
        output = operator.operator(label=label, operated_image=gt_boxes)
        gt_box_image = draw_box(output[0], gt_boxes[0])
        utils.save_image(gt_box_image, f'{results_folder}/box_gt_new_img_{batch_ind}_trial_{rep}.png')






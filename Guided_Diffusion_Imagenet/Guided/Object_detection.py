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
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--wd", default=1e-2, type=float)
parser.add_argument("--shuffle", default=False, help='shuffles the data when we can the train and val data')
parser.add_argument('--direct', action='store_true', help='use direct sampling for noising and denoising')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--run_command', default='PYTHONPATH=. python Guided/membership_classification.py',
                    help='How to run the script.')
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--test_every', type=int, default=10)
parser.add_argument('--optimizer', default='adamw', choices=['sgd', 'adamw'])
parser.add_argument('--use_noise', action='store_true')
parser.add_argument('--fixed_noise', action='store_true')
parser.add_argument('--almost_fixed_noise', action='store_true')
parser.add_argument('--distribution', action='store_true')
parser.add_argument('--load', action='store_true')
parser.add_argument('--remove_bn', action='store_true', default=False)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--use_image', type=int, default=1)
parser.add_argument('--wandb', type=int, default=1)
parser.add_argument('--input_size', type=int, default=64)
parser.add_argument("--optim_lr", default=1e-3, type=float)
parser.add_argument('--optim_max_iters', type=int, default=1)
parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
parser.add_argument('--optim_guidance_3', action='store_true', default=False)
parser.add_argument('--optim_original_guidance', action='store_true', default=False)
parser.add_argument("--optim_guidance_3_wt", default=2.0, type=float)
parser.add_argument("--optim_guidance_test", action='store_true')
parser.add_argument("--optim_guidance_test_wt", default=2.0, type=float)
parser.add_argument('--optim_warm_start', action='store_true', default=False)
parser.add_argument('--optim_print', action='store_true', default=False)
parser.add_argument('--optim_aug', action='store_true', default=False)
parser.add_argument('--optim_folder', default='/cmlscratch/hmchu/guided-diffusion/')
parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
parser.add_argument("--optim_mask_fraction", default=0.5, type=float)
parser.add_argument("--exp_name", default='')
parser.add_argument("--split", default=1, type=int)

args = parser.parse_args()

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# PYTHONPATH=. python scripts/load_model.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# PYTHONPATH=. python Guided/Resnet_guided.py $MODEL_FLAGS --classifier_scale 0.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS --batch_size 4

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# PYTHONPATH=. python Guided/Resnet_guided.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS --batch_size 4


BATCH_SIZE = args.batch_size
resolution_fact = 8

model = models.resnet18(pretrained=True)
Trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import pickle

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

def ce_loss(input, target):
    c = nn.CrossEntropyLoss(reduce=False)
    l = c(input, target)
    return l


# operation = [2, operation_func, optim.Adam, 0.008, weighted_ce_loss, 500, 0.01, 1]
# operation = [2, operation_func, optim.Adam, 0.01, mse_loss, 2000, 0.005, 1]

results_folder = os.path.join(args.optim_folder, args.exp_name)
create_folder(results_folder)

operation = OptimizerDetails()


operation.num_steps = args.optim_num_steps #[2]
operation.operation_func = operation_func
operation.optimizer = 'Adam'
operation.lr = args.optim_lr #0.01
operation.loss_func = ce_loss
operation.max_iters = args.optim_max_iters #00
operation.loss_cutoff = args.optim_loss_cutoff #0.00001
operation.tv_loss = None
operation.guidance_3 = args.optim_guidance_3 #True
operation.guidance_test = args.optim_guidance_test 
operation.original_guidance = args.optim_original_guidance
operation.optim_guidance_3_wt = args.optim_guidance_3_wt
operation.optim_guidance_test_wt = args.optim_guidance_test_wt
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
train_dataset, val_dataset = get_train_val_datasets(args)
print('done')
print('splitting the dataset...')
train1, train2 = get_splitted_dataset(dataset=train_dataset,
                                      checkpoint_path='/cmlscratch/bansal01/fall_2022/Guided_Diffusion_Imagenet/checkpoints//non_equal_split/partitions_train.pt')
val1, val2 = get_splitted_dataset(dataset=val_dataset, checkpoint_path='/cmlscratch/bansal01/fall_2022/Guided_Diffusion_Imagenet/checkpoints//non_equal_split/partitions_val.pt')
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
    #box = box * 2 - 1
    return box

def save_box(pred, path):
     pred_cpu = []
     for item in pred:
         item_cpu = {}
         for k in item:
             item_cpu[k] = item[k].cpu()
         pred_cpu.append(item_cpu)
     with open(path, 'wb') as f:
         pickle.dump(pred_cpu, f)





for batch_ind, batch in enumerate(val1):
    image, label = batch
    image, label = image.cuda(), label.cuda()
    
    if args.split == 1:
        if batch_ind not in [2, 9, 16, 24]:
            continue
    elif args.split == 2:
        if batch_ind not in [68, 164, 238]:
            continue
    else:
        raise

    with torch.no_grad():
        map = label

    gt_boxes = operation_func(image)
    save_box(gt_boxes, f'{results_folder}/box_original_batch_{batch_ind}')

    utils.save_image((image + 1) * 0.5, f'{results_folder}/og_img_{batch_ind}.png')
    for b in range(BATCH_SIZE):
        box_image = draw_box(image[b], gt_boxes[b])
        utils.save_image(box_image, f'{results_folder}/box_og_img_{batch_ind}_{b}.png')

    for rep in range(10):
        output = operator.operator(label=label, operated_image=gt_boxes)
        pred_boxes = operation_func(output)
        for b in range(BATCH_SIZE):
            gt_box_image = draw_box(output[b], gt_boxes[b])
            pred_box_image = draw_box(output[b], pred_boxes[b])
            utils.save_image(gt_box_image, f'{results_folder}/box_gt_new_img_{batch_ind}_{b}_trial_{rep}.png')
            utils.save_image(pred_box_image, f'{results_folder}/box_pred_new_img_{batch_ind}_{b}_trial_{rep}.png')

        try:
            save_box(pred_boxes, f'{results_folder}/box_pred_batch_{batch_ind}_trial_{rep}')
        except:
            pass




    if batch_ind == 300:
        break





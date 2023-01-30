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
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, deeplabv3_resnet101, lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import random
from typing import Any, Callable, List, Optional, Tuple
import torch.optim as optim
from torch import nn, einsum
from torch.autograd import Variable
import torchgeometry as tgm

import torch
import torch.utils.data

from Guided.dataset.helpers import get_splitted_dataset
from Guided.helpers import get_parser, Operation, OptimizerDetails
from Guided.models.resnet import ResNet18_64x64, ResNet18_64x64_1, ResNet18_256x256
from scripts.imagenet import get_loader_from_dataset, get_train_val_datasets
import cv2

import torchvision
import cv2
from torchvision import transforms, utils
from torch.utils import data
import torch.nn.functional as F
import os
import errno
import shutil

def create_folder(path):
    try:
        os.mkdir(path)
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

parser.add_argument("--optim_lr", default=1e-2, type=float)
parser.add_argument('--optim_max_iters', type=int, default=1)
parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
parser.add_argument('--optim_guidance_3', action='store_true', default=False)
parser.add_argument('--optim_original_guidance', action='store_true', default=False)
parser.add_argument("--optim_guidance_3_wt", default=2.0, type=float)
parser.add_argument("--optim_tv_loss", default=None, type=float)
parser.add_argument('--optim_warm_start', action='store_true', default=False)
parser.add_argument('--optim_print', action='store_true', default=False)
parser.add_argument('--optim_folder', default='./temp/')
parser.add_argument('--optim_sampling_type', default=None)
parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
parser.add_argument('--blur_size', type=int, default=11)
parser.add_argument('--blur_std', type=float, default=3)


args = parser.parse_args()

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# PYTHONPATH=. python scripts/load_model.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# PYTHONPATH=. python Guided/Segmentation_mobilenet.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS --batch_size 4



BATCH_SIZE = args.batch_size

def mse_loss(input, target):
    return ((input - target) ** 2).mean(dim=[1, 2, 3])

def blur(dims, std):
    return tgm.image.get_gaussian_kernel2d(dims, std)

def get_conv(dims, std, mode='reflect'):
    kernel = blur(dims, std)
    conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=dims,
                     padding=int((dims[0] - 1) / 2), padding_mode=mode,
                     bias=False, groups=3)
    with torch.no_grad():
        kernel = torch.unsqueeze(kernel, 0)
        kernel = torch.unsqueeze(kernel, 0)
        kernel = kernel.repeat(3, 1, 1, 1)
        conv.weight = nn.Parameter(kernel)

    return conv

conv = get_conv((args.blur_size, args.blur_size), (args.blur_std, args.blur_std)).cuda()
conv.eval()


results_folder = args.optim_folder
create_folder(results_folder)

operation = OptimizerDetails()

operation.num_steps = args.optim_num_steps #[2]

operation.optimizer = 'Adam'
operation.lr = args.optim_lr
operation.loss_func = mse_loss

operation.max_iters = args.optim_max_iters
operation.loss_cutoff = args.optim_loss_cutoff
operation.tv_loss = args.optim_tv_loss
operation.operation_func = conv


operation.guidance_3 = args.optim_guidance_3
operation.original_guidance = args.optim_original_guidance
operation.mask_type = args.optim_mask_type

operation.optim_guidance_3_wt = args.optim_guidance_3_wt
operation.do_guidance_3_norm = args.optim_do_guidance_3_norm
operation.optim_unscaled_guidance_3 = args.optim_unscaled_guidance_3
operation.sampling_type = args.optim_sampling_type

operation.warm_start = args.optim_warm_start
operation.print = args.optim_print
operation.print_every = 50
operation.folder = results_folder


# operation = [2, operation_func, optim.Adam, 0.001, nn.MSELoss(), 1000, 0.001]

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
                                      checkpoint_path='checkpoints/non_equal_split/partitions_train.pt')
val1, val2 = get_splitted_dataset(dataset=val_dataset, checkpoint_path='checkpoints/non_equal_split/partitions_val.pt')
print('done')
train1, train2 = get_loader_from_dataset(args, train1, True), get_loader_from_dataset(args, train2, False)
val1, val2 = get_loader_from_dataset(args, val1, True), get_loader_from_dataset(args, val2, False)


for batch_ind, batch1 in enumerate(val1):

    image, label = batch1
    image, label = image.cuda(), label.cuda()
    print(label)

    utils.save_image((image + 1) * 0.5, f'{results_folder}/og_img_{batch_ind}.png')

    with torch.no_grad():
        blur_image = operation.operation_func(image)

    utils.save_image((blur_image + 1) * 0.5, f'{results_folder}/image_blur_{batch_ind}.png')


    print("Start")
    output = operator.operator(label=label, operated_image=blur_image)
    utils.save_image((output + 1) * 0.5, f'{results_folder}/new_img_{batch_ind}.png')


    if batch_ind == 0:
        break




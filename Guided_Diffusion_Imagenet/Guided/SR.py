from pycocotools.coco import COCO
import numpy as np
import random
import os
import cv2

### For visualizing the outputs ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io, transforms, utils
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
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

torch.manual_seed(0)

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

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
parser.add_argument('--resolution_fact', type=int, default=4)
parser.add_argument('--num_steps', type=int, default=5)
parser.add_argument('--save_folder', type=str, default=None)

args = parser.parse_args()

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# PYTHONPATH=. python scripts/load_model.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# PYTHONPATH=. python Guided/SR.py $MODEL_FLAGS --classifier_scale 0.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt $SAMPLE_FLAGS --resolution_fact 8 --batch_size 16 --num_steps 3



BATCH_SIZE = args.batch_size
resolution_fact = args.resolution_fact

operation_func = torch.nn.AvgPool2d(resolution_fact)
m = torch.nn.Upsample(scale_factor=resolution_fact, mode='nearest')

operation_func = torch.nn.DataParallel(operation_func).cuda()
operation_func.eval()


def mse_loss(input, target):
    return ((input - target) ** 2).mean(dim=[1, 2, 3])

# operation = [5, operation_func, optim.Adam, 0.001, mse_loss, 1000, 0.001]

operation = OptimizerDetails()

operation.num_steps = [args.num_steps]
operation.operation_func = operation_func
operation.optimizer = 'Adam'
operation.lr = 0.001
operation.loss_func = mse_loss
operation.max_iters = 1000
operation.loss_cutoff = 0.0001
operation.lr_scheduler = 'CosineAnnealingLR'


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

if args.save_folder == None:
    results_folder = f'./SR_{resolution_fact}_{args.num_steps}/'
else:
    results_folder = args.save_folder

create_folder(results_folder)

cnt = 0
for batch_ind, batch in enumerate(val1):
    image, label = batch
    image, label = image.cuda(), label.cuda()

    with torch.no_grad():
        lr_img = operation_func(image)
    output = operator.operator(label=label, operated_image=lr_img)

    lr_img = m(lr_img)
    for j in range(image.shape[0]):
        input_ = return_cv2(image[j], f'{results_folder}/og_img_{cnt}.png')
        output_ = return_cv2(output[j], f'{results_folder}/output_{cnt}.png')
        lr_img_ = return_cv2(lr_img[j], f'{results_folder}/lr_img_{cnt}.png')

        im = cv2.hconcat([input_, lr_img_, output_])
        cv2.imwrite(f'{results_folder}/all_{cnt}.png', im)

        cnt+=1


    if batch_ind == 0:
        break





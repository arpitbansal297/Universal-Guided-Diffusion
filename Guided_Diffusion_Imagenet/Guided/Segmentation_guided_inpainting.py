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

import torch
import torch.utils.data

from Guided.dataset.helpers import get_splitted_dataset
from Guided.helpers import get_parser, Operation, OptimizerDetails, Evaluator
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
parser.add_argument('--optim_backward_guidance_max_iters', type=int, default=1)
parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
parser.add_argument('--optim_original_guidance', action='store_true', default=False)
parser.add_argument("--optim_forward_guidance_wt", default=2.0, type=float)
parser.add_argument("--optim_tv_loss", default=None, type=float)
parser.add_argument('--optim_warm_start', action='store_true', default=False)
parser.add_argument('--optim_print', action='store_true', default=False)
parser.add_argument('--optim_folder', default='./temp/')
parser.add_argument('--optim_sampling_type', default=None)
parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
parser.add_argument('--batches', type=int, default=1)
parser.add_argument('--trials', type=int, default=1)

args = parser.parse_args()

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# PYTHONPATH=. python scripts/load_model.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# PYTHONPATH=. python Guided/Segmentation_mobilenet.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS --batch_size 4



BATCH_SIZE = args.batch_size
resolution_fact = 8

weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT# DeepLabV3_ResNet101_Weights.DEFAULT #FCN_ResNet50_Weights.DEFAULT
model = lraspp_mobilenet_v3_large(weights=weights)# deeplabv3_resnet101(weights=weights) #fcn_resnet50(weights=weights)
Trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = model.eval()
for param in model.parameters():
    param.requires_grad = False


class Segmnetation(nn.Module):
    def __init__(self, model, Trans):
        super(Segmnetation, self).__init__()
        self.model = model#.backbone
        self.trans = Trans

    def forward(self, x):
        map = (x + 1) * 0.5
        map = TF.resize(map, (520, 520), interpolation=TF.InterpolationMode.BILINEAR)
        map = self.trans(map)
        map = self.model(map)
        map = map['out']
        return map

operation_func = Segmnetation(model, Trans)
operation_func = torch.nn.DataParallel(operation_func).cuda()
operation_func.eval()
for param in operation_func.parameters():
    param.requires_grad = False


def mse_loss(input, target):
    actual_target = target[0]
    mask = target[1]

    m_sum = mask.sum(dim=[1, 2, 3])
    m_sum = mask.shape[1] * mask.shape[2] / m_sum

    input = input * mask
    loss = (input - actual_target) ** 2
    return loss.mean(dim=[1, 2, 3]) * m_sum

def CrossEntropyLoss(logit, target):
    target = target[-1]
    criterion = nn.CrossEntropyLoss(reduce=False, ignore_index=255).cuda()
    loss = criterion(logit, target.long())

    return loss.mean(dim=[1, 2])

results_folder = args.optim_folder
create_folder(results_folder)

operation = OptimizerDetails()

seq = []
pre = torch.nn.Sequential(*seq)

operation.num_steps = args.optim_num_steps #[2]
operation.operation_func = None
operation.other_guidance_func = operation_func

operation.optimizer = 'Adam'
operation.lr = args.optim_lr #0.01
operation.loss_func = mse_loss
operation.other_criterion = CrossEntropyLoss

operation.max_iters = args.optim_backward_guidance_max_iters #00
operation.loss_cutoff = args.optim_loss_cutoff #0.00001
operation.tv_loss = args.optim_tv_loss

operation.guidance_3 = args.optim_forward_guidance #True
operation.original_guidance = args.optim_original_guidance
operation.mask_type = args.optim_mask_type

operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
operation.do_guidance_3_norm = args.optim_do_forward_guidance_norm
operation.optim_unscaled_guidance_3 = args.optim_unscaled_forward_guidance
operation.sampling_type = args.optim_sampling_type

operation.warm_start = args.optim_warm_start #False
operation.print = args.optim_print
operation.print_every = 10
operation.folder = results_folder
if args.optim_aug:
    operation.Aug = pre


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

def read_cv2(img, path):
    black = [255, 255, 255]
    utils.save_image(img, path, nrow=1)
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
    return img


train_dataset, val_dataset = get_train_val_datasets(args)
val1, _ = get_splitted_dataset(dataset=val_dataset, checkpoint_path='checkpoints/partitions_val.pt')
val1 = get_loader_from_dataset(args, val1, True)

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [
                           0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [
                           64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])

def decode_seg_map_sequence(label_masks):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask):
    n_classes = 21
    label_colours = get_pascal_labels()

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()

    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def get_images():
    take_labels = [i for i in range(153,260)]

    dog_images = []
    dog_labels = []

    for batch_ind, batch in enumerate(val1):
        image, label = batch
        for i in range(label.shape[0]):
            if label[i] in take_labels:
                dog_images.append(image[i:i+1])
                dog_labels.append(label[i:i+1])

                if len(dog_images) == BATCH_SIZE * args.batches:
                    return dog_images, dog_labels


dog_images, dog_labels = get_images()
dog_images = torch.concat(dog_images, dim=0)
dog_labels = torch.concat(dog_labels, dim=0)

print("All")
print(dog_images.shape)

# evaluator = Evaluator(21)

for batch_ind in range(args.batches):

    start = batch_ind * BATCH_SIZE
    end = batch_ind * BATCH_SIZE + BATCH_SIZE

    print(start, end)

    image, label = dog_images[start: end], dog_labels[start: end]
    image, label = image.cuda(), label.cuda()
    print(label)

    with torch.no_grad():
        map = operation_func(image).softmax(dim=1)

        target_np = map.data.cpu().numpy()
        target_np = np.argmax(target_np, axis=1)

        old_map = torch.clone(map)
        num_class = map.shape[1]
        print(map.shape)
        #
        max_vals, max_indices = torch.max(map, 1)
        print(max_indices.shape)
        map = max_indices

        sep_map = F.one_hot(map, num_classes=num_class)
        sep_map = sep_map.permute(0, 3, 1, 2).float()
        print(sep_map.shape)

    label_save = decode_seg_map_sequence(torch.squeeze(map, 1).detach(
    ).cpu().numpy())

    utils.save_image(label_save, f'{results_folder}/label_{batch_ind}.png')
    utils.save_image((image + 1) * 0.5, f'{results_folder}/og_img_{batch_ind}.png')

    mask = sep_map[:, 0:1, :, :]
    mask = TF.resize(mask, (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
    operator.operation.guidance_mask = 1 - mask
    image_mask = image * mask

    utils.save_image(mask, f'{results_folder}/mask_{batch_ind}.png')
    utils.save_image((image_mask + 1) * 0.5, f'{results_folder}/image_mask_{batch_ind}.png')

    for trials in range(args.trials):
        print("Start")
        output = operator.operator(label=label, operated_image=[image_mask, mask, map])
        utils.save_image((output + 1) * 0.5, f'{results_folder}/new_img_{batch_ind}_trial_{trials}.png')


        with torch.no_grad():
            new_map = operation_func(output)

            pred = new_map.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            new_image_map = new_map.softmax(dim=1)
            num_class = new_map.shape[1]

            max_vals, max_indices = torch.max(new_image_map, 1)
            new_image_map = max_indices

        new_image_map_save = decode_seg_map_sequence(torch.squeeze(new_image_map, 1).detach(
        ).cpu().numpy())

        utils.save_image(new_image_map_save, f'{results_folder}/new_image_map_save_{batch_ind}_trial_{trials}.png')

        print(target_np.shape, pred.shape)
        # evaluator.add_batch(target_np, pred)

        for i in range(image.shape[0]):
            og_img_ = return_cv2(image[i], f'{results_folder}/og_img_trial_{trials}.png')
            label_save_ = read_cv2(label_save[i], f'{results_folder}/label_save_trial_{trials}.png')

            output_ = return_cv2(output[i], f'{results_folder}/img.png')
            new_image_map_save_ = read_cv2(new_image_map_save[i], f'{results_folder}/direct_recons_trial_{trials}.png')

            im_l = cv2.hconcat([og_img_, output_])
            im_h = cv2.hconcat([label_save_, new_image_map_save_])

            cv2.imwrite(f'{results_folder}/images_{cnt}_trial_{trials}.png', im_l)
            cv2.imwrite(f'{results_folder}/map_{cnt}_trial_{trials}.png', im_h)

            cnt += 1


# Acc = evaluator.Pixel_Accuracy()
# Acc_class = evaluator.Pixel_Accuracy_Class()
# mIoU = evaluator.Mean_Intersection_over_Union()
# FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
#
# print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))


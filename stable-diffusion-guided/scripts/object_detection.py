import sys
import os
from pathlib import Path
import os
import errno

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

print(sys.path)


import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext


from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad

from transformers import AutoFeatureExtractor

from torchvision import transforms, utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.optim import SGD, Adam, AdamW
import PIL
from torch.utils import data
from pathlib import Path
from PIL import Image
from torchvision import transforms, utils
import random
from scripts.helper import OptimizerDetails
import clip
import os
import inspect
import torchvision.transforms.functional as TF

import pickle




def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def return_cv2(img, path):
    black = [255, 255, 255]
    img = (img + 1) * 0.5
    utils.save_image(img, path, nrow=1)
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
    return img

def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

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
        inter = self.preprocess((x + 1) * 0.5)
        return self.model(inter)
    
    def cal_loss(self, x, gt):
        def set_bn_to_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()


        self.model.train()
        self.model.backbone.eval()
        self.model.apply(set_bn_to_eval)
        inter = self.preprocess((x + 1) * 0.5)
        loss = self.model(inter, gt)
        return loss['loss_classifier'] + loss['loss_objectness'] + loss['loss_rpn_box_reg']
        




def get_optimation_details(args):

    guidance_func = ObjectDetection().cuda()
    operation = OptimizerDetails()

    operation.num_steps = args.optim_num_steps
    operation.operation_func = guidance_func

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = None

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff

    operation.guidance_3 = args.optim_forward_guidance
    operation.guidance_2 = args.optim_backward_guidance

    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
    operation.original_guidance = args.optim_original_conditioning

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 5
    operation.folder = args.optim_folder

    return operation

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument("--optim_lr", default=1e-2, type=float)
    parser.add_argument('--optim_max_iters', type=int, default=1)
    parser.add_argument('--optim_mask_type', type=int, default=1)
    parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
    parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_backward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_original_conditioning', action='store_true', default=False)
    parser.add_argument("--optim_forward_guidance_wt", default=5.0, type=float)
    parser.add_argument('--optim_do_forward_guidance_norm', action='store_true', default=False)
    parser.add_argument("--optim_tv_loss", default=None, type=float)
    parser.add_argument('--optim_warm_start', action='store_true', default=False)
    parser.add_argument('--optim_print', action='store_true', default=False)
    parser.add_argument('--optim_aug', action='store_true', default=False)
    parser.add_argument('--optim_folder', default='./temp/')
    parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
    parser.add_argument("--optim_mask_fraction", default=0.5, type=float)
    parser.add_argument('--text', default=None)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--indexes", nargs="+", default=[0, 1, 2], type=int)

    opt = parser.parse_args()

    results_folder = opt.optim_folder
    create_folder(results_folder)

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()


    sampler = DDIMSamplerWithGrad(model)

    batch_size = opt.n_samples
    assert batch_size == 1

    operation = get_optimation_details(opt)

    torch.set_grad_enabled(False)

    if opt.text is not None:
        prompt = opt.text
    else:
        prompt = "a headshot of a woman with a dog"


    print(prompt)
    
    # Setup for object detection


    def gen_box(num_image, anchor_locs, label, sizes):

        objd_cond = []
        for _ in range(num_image):
            boxes = []
            labels = torch.Tensor(label).long()
            for aidx, anchor_loc in enumerate(anchor_locs):
                x, y = anchor_loc
                size = sizes[aidx]
                if isinstance(size, list):
                    x_size = size[0]
                    y_size = size[1]
                else:
                    x_size = size
                    y_size = size
                box = [x - x_size / 2, y - y_size/2, x + x_size/2, y + y_size/2]
                boxes.append(box)
            boxes = torch.Tensor(boxes)
            objd_cond.append({'boxes': boxes.cuda(), 'labels': labels.cuda()})
        return objd_cond
 
    def draw_box(img, pred):
        labels = [obj_categories[j] for j in pred["labels"].cpu()]
        uint8_image = (img.cpu() * 255).to(torch.uint8)
        box = draw_bounding_boxes(uint8_image, boxes=pred["boxes"].cpu(),
                labels=labels,
                colors="red", width=4)
        box = box.float() / 255.0
        box = box * 2 - 1
        return box
    
    
    for index in opt.indexes:
        
        print(f'current bounding box:{index}')
        # Change the bounding box definition here
        if index == 0:
            obj_det_cats = ["dog", "person"]
            test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
            sizes = [180, [200,400]]
        elif index == 1:
            obj_det_cats = ["person", "dog"]
            test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
            sizes = [[200,400], 180]
        elif index == 2:
            obj_det_cats = ["dog", "person"]
            test_anchor_locs = [(150.0, 384.0), (384.0, 256.0)]
            sizes = [[275,200], [200,400]]
    
    
        obj_categories = operation.operation_func.categories
        category = [obj_categories.index(cc) for cc in obj_det_cats]
        og_img_guide = gen_box(opt.n_samples, test_anchor_locs, category, sizes)
    
        tic = time.time()
        all_samples = list()
    
        for n in trange(opt.trials, desc="Sampling"):
 
            uc = None
            if opt.scale != 1.0:
                uc = model.module.get_learned_conditioning(batch_size * [""])
            c = model.module.get_learned_conditioning([prompt])
    
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            samples_ddim = sampler.sample_seperate(S=opt.ddim_steps,
                                            conditioning=c,
                                            batch_size=opt.n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=opt.scale,
                                            unconditional_conditioning=uc,
                                            eta=opt.ddim_eta,
                                            operated_image=og_img_guide,
                                            operation=operation)
    
            x_samples_ddim = model.module.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    
            utils.save_image(x_samples_ddim, f'{results_folder}/new_img_{n}.png')
            box_original_output = draw_box(x_samples_ddim[0].detach(), og_img_guide[0])
            img_ = return_cv2(box_original_output, f'{results_folder}/box_new_img_{n}.png')


if __name__ == "__main__":
    main()

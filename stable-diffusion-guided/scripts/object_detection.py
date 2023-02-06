import sys
import os
from pathlib import Path

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

print(sys.path)


import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
#from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext


from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
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
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.datasets import ImageFolder

import pickle



# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


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


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, data_aug=False, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        random.shuffle(self.paths)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((self.image_size, self.image_size), resample=PIL.Image.LANCZOS)

        return self.transform(img)

def return_cv2(img, path):
    black = [255, 255, 255]
    img = (img + 1) * 0.5
    utils.save_image(img, path, nrow=1)
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
    return img

def cycle(dl):
    while True:
        for data in dl:
            yield data

import os
import errno
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
        




def cycle_cat(dl):
    while True:
        for data in dl:
            yield data[0]

def l1_loss(input, target):
    l = torch.abs(input - target).mean(dim=[1])
    return l

def get_optimation_details(args):
    mtcnn_face = not args.center_face
    print('mtcnn_face')
    print(mtcnn_face)

    guidance_func = ObjectDetection().cuda()
    operation = OptimizerDetails()

    operation.num_steps = args.optim_num_steps
    operation.operation_func = guidance_func

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = l1_loss

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff

    operation.guidance_3 = args.optim_guidance_3
    operation.guidance_2 = args.optim_guidance_2

    operation.optim_guidance_3_wt = args.optim_guidance_3_wt
    operation.original_guidance = args.optim_original_guidance

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 5
    operation.folder = args.optim_folder

    return operation

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
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
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
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
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument("--optim_lr", default=1e-2, type=float)
    parser.add_argument('--optim_max_iters', type=int, default=1)
    parser.add_argument('--optim_mask_type', type=int, default=1)
    parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
    parser.add_argument('--optim_guidance_3', action='store_true', default=False)
    parser.add_argument('--optim_guidance_2', action='store_true', default=False)
    parser.add_argument('--optim_original_guidance', action='store_true', default=False)
    parser.add_argument("--optim_guidance_3_wt", default=5.0, type=float)
    parser.add_argument('--optim_do_guidance_3_norm', action='store_true', default=False)
    parser.add_argument("--optim_tv_loss", default=None, type=float)
    parser.add_argument('--optim_warm_start', action='store_true', default=False)
    parser.add_argument('--optim_print', action='store_true', default=False)
    parser.add_argument('--optim_aug', action='store_true', default=False)
    parser.add_argument('--optim_folder', default='./temp/')
    parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
    parser.add_argument("--optim_mask_fraction", default=0.5, type=float)
    parser.add_argument("--text", default=None)
    parser.add_argument('--text_type', type=int, default=1)
    parser.add_argument("--batches", default=0, type=int)

    parser.add_argument('--fr_crop', action='store_true')
    parser.add_argument('--center_face', action='store_true')
    parser.add_argument('--big_text', action='store_true')


    opt = parser.parse_args()

    results_folder = opt.optim_folder
    create_folder(results_folder)

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()


    sampler = DDIMSamplerWithGrad(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    #wm = "StableDiffusionV1"
    #wm_encoder = WatermarkEncoder()
    #wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1



    operation = get_optimation_details(opt)

    image_size = 256
    transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    ds = ImageFolder(root='/cmlscratch/hmchu/datasets/celeba_hq_256/', transform=transform)
    dl = cycle_cat(data.DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16,
                                   drop_last=True))

    torch.set_grad_enabled(False)

    if opt.text_type == 1:
        prompt = "a photograph of two cats near a river"
        obj_det_cats = ["cat", "cat"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [170, 170]
    elif opt.text_type == 2:
        prompt = "a photograph of a cat and a dog near a river"
        obj_det_cats = ["cat", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [170, 170]
    elif opt.text_type == 3:
        prompt = "an oil painting of two cats near a river"
        obj_det_cats = ["cat", "cat"]
        test_anchor_locs = [(128.0, 128.0), (384.0, 384.0)]
        sizes = [200, 140]
    elif opt.text_type == 4:
        prompt = "an oil painting of a cat and a dog near a river"
        obj_det_cats = ["cat", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [200, 140]
    elif opt.text_type == 5:
        prompt = "an oil painting of a cat and a dog near a river"
        obj_det_cats = ["cat", "dog"]
        test_anchor_locs = [(128.0, 128.0), (384.0, 384.0)]
        sizes = [170, 170]
    elif opt.text_type == 6:
        prompt = "a headshot of a woman with a dog"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,400]]
    elif opt.text_type == 7:
        prompt = "a headshot of a man with a dog"
        obj_det_cats = ["person", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [[200, 400], 180]
    elif opt.text_type == 8:
        prompt = "two chairs and a person in a room"
        obj_det_cats = ["chair", "person", "chair"]
        test_anchor_locs = [(100.0, 400.0), (256.0, 256.0), (400.0, 256.0)]
        sizes = [140, [150, 400], [200, 200]]
    elif opt.text_type == 9:
        prompt = "an oil painting of a headshot of a woman with a dog"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,400]]
    elif opt.text_type == 10:
        prompt = "a headshot of a woman with a dog in space"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,400]]
    elif opt.text_type == 11:
        prompt = "a headshot of a woman with a dog on beach"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,400]]
    elif opt.text_type == 12:
        prompt = "a headshot of a woman with a dog in winter"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,400]]
    elif opt.text_type == 13:
        prompt = "a headshot of a woman with a dog on new york street"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,400]]
    elif opt.text_type == 14:
        prompt = "a headshot of a woman with a dog in space"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,400]]
    elif opt.text_type == 15:
        prompt = "a headshot of a woman with a dog made of marble"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,400]]
    elif opt.text_type == 16:
        prompt = "a headshot of a woman made of marble with a dog"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,400]]

    elif opt.text_type == 17:
        prompt = "a headshot of a woman with a dog"
        obj_det_cats = ["person", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,350]]
    elif opt.text_type == 18:
        prompt = "a headshot of a woman with a dog"
        obj_det_cats = ["person", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [[200,400], 180]
    elif opt.text_type == 19:
        prompt = "a headshot of a woman with a dog"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(150.0, 384.0), (384.0, 256.0)]
        sizes = [[275,200], [200,400]]
    elif opt.text_type == 20:
        prompt = "a headshot of a woman with a dog"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(150.0, 128.0), (384.0, 256.0)]
        sizes = [[275,200], [200,400]]

    elif opt.text_type == 21:
        prompt = "a headshot of a woman with a dog in space"
        obj_det_cats = ["person", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,350]]
    elif opt.text_type == 22:
        prompt = "a headshot of a woman with a dog in space"
        obj_det_cats = ["person", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [[200,400], 180]
    elif opt.text_type == 23:
        prompt = "a headshot of a woman with a dog in space"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(150.0, 384.0), (384.0, 256.0)]
        sizes = [[275,200], [200,400]]
    elif opt.text_type == 24:
        prompt = "a headshot of a woman with a dog in space"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(150.0, 128.0), (384.0, 256.0)]
        sizes = [[275,200], [200,400]]

    elif opt.text_type == 25:
        prompt = "a headshot of a woman with a dog on beach"
        obj_det_cats = ["person", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,350]]
    elif opt.text_type == 26:
        prompt = "a headshot of a woman with a dog on beach"
        obj_det_cats = ["person", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [[200,400], 180]
    elif opt.text_type == 27:
        prompt = "a headshot of a woman with a dog on beach"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(150.0, 384.0), (384.0, 256.0)]
        sizes = [[275,200], [200,400]]
    elif opt.text_type == 28:
        prompt = "a headshot of a woman with a dog on beach"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(150.0, 128.0), (384.0, 256.0)]
        sizes = [[275,200], [200,400]]

    elif opt.text_type == 29:
        prompt = "an oil painting of a headshot of a woman with a dog"
        obj_det_cats = ["person", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,350]]
    elif opt.text_type == 30:
        prompt = "an oil painting of a headshot of a woman with a dog"
        obj_det_cats = ["person", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [[200,400], 180]
    elif opt.text_type == 31:
        prompt = "an oil painting of a headshot of a woman with a dog"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(150.0, 384.0), (384.0, 256.0)]
        sizes = [[275,200], [200,400]]
    elif opt.text_type == 32:
        prompt = "an oil painting of a headshot of a woman with a dog"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(150.0, 128.0), (384.0, 256.0)]
        sizes = [[275,200], [200,400]]

    elif opt.text_type == 33:
        prompt = "a headshot of a woman with a dog on new york street"
        obj_det_cats = ["person", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [180, [200,350]]
    elif opt.text_type == 34:
        prompt = "a headshot of a woman with a dog on new york street"
        obj_det_cats = ["person", "dog"]
        test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
        sizes = [[200,400], 180]
    elif opt.text_type == 35:
        prompt = "a headshot of a woman with a dog on new york street"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(150.0, 384.0), (384.0, 256.0)]
        sizes = [[275,200], [200,400]]
    elif opt.text_type == 36:
        prompt = "a headshot of a woman with a dog on new york street"
        obj_det_cats = ["dog", "person"]
        test_anchor_locs = [(150.0, 128.0), (384.0, 256.0)]
        sizes = [[275,200], [200,400]]





    print(prompt)
    
    # Setup for object detection

    obj_categories = operation.operation_func.categories
    category = [obj_categories.index(cc) for cc in obj_det_cats]

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
        print(obj_categories)
        labels = [obj_categories[j] for j in pred["labels"].cpu()]
        uint8_image = (img.cpu() * 255).to(torch.uint8)
        font_size = 10
        if opt.big_text:
            font_size = 50
        box = draw_bounding_boxes(uint8_image, boxes=pred["boxes"].cpu(),
                labels=labels,
                colors="red", font="/usr/share/fonts/google-crosextra-carlito/Carlito-Regular.ttf",
                width=4, font_size=font_size)
        box = box.float() / 255.0
        box = box * 2 - 1
        return box


    og_img_guide = gen_box(opt.n_samples, test_anchor_locs, category, sizes)



    tic = time.time()
    all_samples = list()

    for n in trange(opt.n_iter, desc="Sampling"):

        uc = None
        if opt.scale != 1.0:
            uc = model.module.get_learned_conditioning(batch_size * [""])
        c = model.module.get_learned_conditioning([prompt])
        for multiple_tries in range(1):
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

            box_pred = operation.operation_func(x_samples_ddim)
            
            def get_black_image(img):
                img_new = img.clone()
                img_new[:] = -1.0
                return img_new

            utils.save_image(x_samples_ddim, f'{results_folder}/new_img_{n}_{multiple_tries}.png')
            for b in range(opt.n_samples):
                box_original_output = draw_box(x_samples_ddim[b].detach(), og_img_guide[b])
                box_pred_output = draw_box(x_samples_ddim[b].detach(), box_pred[b])

                black_background_box = draw_box(get_black_image(x_samples_ddim[b].detach()), og_img_guide[b])
                img_ = return_cv2(box_original_output, f'{results_folder}/box_new_img_batch_{n}_index_{b}_try_{multiple_tries}.png')
                img_ = return_cv2(box_pred_output, f'{results_folder}/pred_box_new_img_batch_{n}_index_{b}_try_{multiple_tries}.png')
                img_ = return_cv2(black_background_box, f'{results_folder}/black_box_new_img_batch_{n}_index_{b}_try_{multiple_tries}.png')

            def save_box(pred, path):
                pred_cpu = []
                for item in pred:
                    item_cpu = {}
                    for k in item:
                        item_cpu[k] = item[k].cpu()
                    pred_cpu.append(item_cpu)
                with open(path, 'wb') as f:
                    pickle.dump(pred_cpu, f)

            save_box(og_img_guide, f'{results_folder}/box_original_batch_{n}')
            try:
                save_box(box_pred, f'{results_folder}/box_pred_batch_{n}')
            except:
                pass







if __name__ == "__main__":
    main()

import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
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
from helper import OptimizerDetails
import clip
import os
import inspect
import torchvision.transforms.functional as TF



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

torch.manual_seed(0)

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
    print(config.model)
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
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

class Clip(nn.Module):
    def __init__(self, model):
        super(Clip, self).__init__()
        self.model = model
        self.trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, y):

        x = (x + 1) * 0.5
        x = TF.resize(x, (224, 224), interpolation=TF.InterpolationMode.BICUBIC)
        x = self.trans(x)

        logits_per_image, logits_per_text = self.model(x, y)
        return -1 * logits_per_image


def get_optimation_details(args):
    clip_model, clip_preprocess = clip.load("RN50")
    print(clip_preprocess)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    l_func = Clip(clip_model)
    l_func.eval()
    for param in l_func.parameters():
        param.requires_grad = False
    l_func = torch.nn.DataParallel(l_func).cuda()


    operation = OptimizerDetails()

    operation.num_steps = args.optim_num_steps
    operation.operation_func = None
    operation.other_guidance_func = None

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = l_func
    operation.other_criterion = None

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff
    operation.tv_loss = args.optim_tv_loss

    operation.guidance_3 = args.optim_forward_guidance
    operation.guidance_2 = args.optim_backward_guidance

    operation.original_guidance = args.optim_original_conditioning
    operation.mask_type = args.optim_mask_type

    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
    operation.do_guidance_3_norm = args.optim_do_forward_guidance_norm

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 500
    operation.folder = args.optim_folder

    return operation, l_func

def main():
    parser = argparse.ArgumentParser()


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
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/cmlscratch/bansal01/summer_2022/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="epochs",
    )
    parser.add_argument(
        "--save_image_folder",
        type=str,
        default='./sanity_check/',
        help="folder to save",
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
    parser.add_argument("--text", default=None)
    parser.add_argument('--text_type', type=int, default=1)
    parser.add_argument("--batches", default=0, type=int)

    opt = parser.parse_args()

    seed_everything(opt.seed)
    results_folder = opt.optim_folder
    create_folder(results_folder)



    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    # model.requires_grad_(False)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()
    # model.requires_grad_(False)

    operation, l_func = get_optimation_details(opt)


    batch_size = 1
    text = []
    for i in range(batch_size):
        if opt.text != None:
            text.append(opt.text)
        else:
            if opt.text_type == 1:
                text.append("Dog scuba-diving")
            elif opt.text_type == 2:
                text.append("a photograph of an astronaut riding a horse")
            elif opt.text_type == 3:
                text.append("an oil painting of a corgi wearing a party hat")
            elif opt.text_type == 4:
                text.append("a hedgehog using a calculator")
            elif opt.text_type == 5:
                text.append("a green train is coming down the tracks")


    print("Text is ", text)


    text = clip.tokenize(text).cuda()

    root = "/cmlscratch/bansal01/summer_2022/stable-diffusion/improved_aesthetics_6plus/train_data_by_websit_512x512/deviantart/"
    train_ds = Dataset(root, opt.W)


    trainloader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True,
                                            num_workers=1,
                                            drop_last=True)

    cnt =0
    for batch_ind, image in enumerate(trainloader):
        xc = image.shape[0] * [""]
        cond = model.module.get_learned_conditioning(xc)

        # image is just for dimensions
        output = model.module.operation_diffusion(og_img=image, operated_image=text, cond=cond, operation=operation)

        for i in range(image.shape[0]):
            img_ = return_cv2(output[i], f'{results_folder}/out_img_{cnt}.png')
            cnt+=1

        if batch_ind == opt.batches:
            break



if __name__ == "__main__":
    main()

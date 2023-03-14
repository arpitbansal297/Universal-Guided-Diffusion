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
from helper import OptimizerDetails, get_seg_text
import clip
import os
import inspect
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
from torchvision.datasets import ImageFolder

from ldm.data.imagenet_openai import get_loader_from_dataset, get_train_val_datasets
import cv2


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def get_splitted_dataset(dataset, checkpoint_path='checkpoints/partitions.pt'):
    obj = torch.load(checkpoint_path)

    partitions_count = obj['partitions_count']
    partitions = obj['partitions']
    output = []
    for partition_ind in range(partitions_count):
        partition = partitions[partition_ind]
        subset_dataset = Subset(dataset, partition)
        output.append(subset_dataset)

    return output


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
    def __init__(self, folder, image_size, data_aug=False, exts=['jpg', 'jpeg', 'png', 'webp']):
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

def read_cv2(img, path):
    black = [255, 255, 255]
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

def cycle_cat(dl):
    while True:
        for data in dl:
            yield data[0]


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
    parser.add_argument("--text", default=None)
    parser.add_argument('--text_type', type=int, default=1)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument('--seg_folder', default='./data/inverse_data')
    parser.add_argument('--special_prompt', default='Walker hound, Walker foxhound')
    parser.add_argument("--trials", default=1, type=int)
    parser.add_argument('--optim_folder', default='./temp/')



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

    image_size = 512
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    batch_size = opt.batch_size

    ds = ImageFolder(root=opt.seg_folder, transform=transform)
    dl = data.DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16,
                         drop_last=True)

    print(len(dl))

    torch.set_grad_enabled(False)

    if opt.text != None:
        prompt = opt.text
    else:
        prompt = get_seg_text(opt.text_type)

    print(prompt)

    for n, d in enumerate(dl, 0):
        image, _ = d
        image = image.cuda()

        if opt.text != None:
            final_prompt = prompt
        else:
            final_prompt = opt.special_prompt + prompt
        print(final_prompt)

        utils.save_image((image + 1) * 0.5, f'{results_folder}/og_img_{n}.png')
        uc = model.module.get_learned_conditioning(batch_size * [""])
        c = model.module.get_learned_conditioning([final_prompt])

        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
        samples_ddim = sampler.sample_inverse(S=opt.ddim_steps,
                                         conditioning=c,
                                         batch_size=batch_size,
                                         shape=shape,
                                         start_image=image,
                                         verbose=False,
                                         unconditional_guidance_scale=opt.scale,
                                         unconditional_conditioning=uc,
                                         eta=opt.ddim_eta)

        x_samples_ddim = model.module.decode_first_stage(samples_ddim)
        x_samples_ddim_unnorm = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        utils.save_image(x_samples_ddim_unnorm, f'{results_folder}/invert_img_{n}.png')


        samples_ddim = sampler.sample_forward(S=opt.ddim_steps,
                                              conditioning=c,
                                              batch_size=batch_size,
                                              shape=shape,
                                              start_zt=samples_ddim,
                                              verbose=False,
                                              unconditional_guidance_scale=opt.scale,
                                              unconditional_conditioning=uc,
                                              eta=opt.ddim_eta)

        x_samples_ddim = model.module.decode_first_stage(samples_ddim)
        x_samples_ddim_unnorm = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        utils.save_image(x_samples_ddim_unnorm, f'{results_folder}/recons_img_{n}.png')

        a = cv2.imread(f'{results_folder}/og_img_{n}.png')
        b = cv2.imread(f'{results_folder}/invert_img_{n}.png')
        c = cv2.imread(f'{results_folder}/recons_img_{n}.png')

        best_imgs = cv2.hconcat([a, b, c])
        cv2.imwrite(f'{results_folder}/all_{n}.png', best_imgs)





if __name__ == "__main__":
    main()

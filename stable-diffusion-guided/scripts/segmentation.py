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

def mse_loss(input, target):
    actual_target = target[0]
    mask = target[1]

    m_sum = mask.sum(dim=[1, 2, 3])
    m_sum = mask.shape[1] * mask.shape[2] / m_sum

    input = input * mask
    loss = (input - actual_target) ** 2
    return loss.mean(dim=[1, 2, 3]) * m_sum

def CrossEntropyLoss(logit, target):
    target = target
    criterion = nn.CrossEntropyLoss(reduce=False, ignore_index=255).cuda()
    loss = criterion(logit, target.long())

    return loss.mean(dim=[1, 2])

def get_optimation_details(args):

    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = lraspp_mobilenet_v3_large( weights=weights)
    Trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = model.eval()

    for param in model.parameters():
        param.requires_grad = False

    operation_func = Segmnetation(model, Trans)
    operation_func = torch.nn.DataParallel(operation_func).cuda()
    operation_func.eval()
    for param in operation_func.parameters():
        param.requires_grad = False

    operation = OptimizerDetails()

    operation.num_steps = args.optim_num_steps
    operation.operation_func = operation_func

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = CrossEntropyLoss

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff

    operation.guidance_3 = args.optim_forward_guidance
    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
    operation.guidance_2 = args.optim_backward_guidance
    operation.original_guidance = args.optim_original_conditioning

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 5
    operation.folder = args.optim_folder

    return operation

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

    parser.add_argument("--optim_lr", default=1e-2, type=float)
    parser.add_argument('--optim_max_iters', type=int, default=1)
    parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
    parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_backward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_original_conditioning', action='store_true', default=False)
    parser.add_argument("--optim_forward_guidance_wt", default=5.0, type=float)
    parser.add_argument("--optim_tv_loss", default=None, type=float)
    parser.add_argument('--optim_warm_start', action='store_true', default=False)
    parser.add_argument('--optim_print', action='store_true', default=False)
    parser.add_argument('--optim_folder', default='./temp/')
    parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
    parser.add_argument("--text", default=None)
    parser.add_argument('--text_type', type=int, default=1)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument('--seg_folder', default='./data/segmentation_data')
    parser.add_argument('--special_prompt', default='Walker hound, Walker foxhound')
    parser.add_argument("--trials", default=10, type=int)
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

    operation = get_optimation_details(opt)

    image_size = 256
    transform = transforms.Compose([
        transforms.CenterCrop(image_size),
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
        if n in opt.indexes:
            image, _ = d
            image = image.cuda()

            if opt.text != None:
                final_prompt = prompt
            else:
                final_prompt = opt.special_prompt + prompt
            print(final_prompt)

            with torch.no_grad():
                map = operation.operation_func(image).softmax(dim=1)

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

            utils.save_image(label_save, f'{results_folder}/label_{n}.png')
            utils.save_image((image + 1) * 0.5, f'{results_folder}/og_img_{n}.png')


            uc = None
            if opt.scale != 1.0:
                uc = model.module.get_learned_conditioning(batch_size * [""])
            c = model.module.get_learned_conditioning([final_prompt])
            for multiple_tries in range(opt.trials):
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, start_zt = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=batch_size,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 operated_image=map,
                                                 operation=operation)

                x_samples_ddim = model.module.decode_first_stage(samples_ddim)
                x_samples_ddim_unnorm = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                utils.save_image(x_samples_ddim_unnorm, f'{results_folder}/new_img_{n}_{multiple_tries}.png')

                with torch.no_grad():
                    new_map = operation.operation_func(x_samples_ddim)

                    pred = new_map.data.cpu().numpy()
                    pred = np.argmax(pred, axis=1)

                    new_image_map = new_map.softmax(dim=1)
                    num_class = new_map.shape[1]

                    max_vals, max_indices = torch.max(new_image_map, 1)
                    new_image_map = max_indices

                new_image_map_save = decode_seg_map_sequence(torch.squeeze(new_image_map, 1).detach(
                ).cpu().numpy())

                utils.save_image(new_image_map_save, f'{results_folder}/new_image_map_save_{n}_{multiple_tries}.png')

                print(target_np.shape, pred.shape)
                torch.save(start_zt, f'{results_folder}/start_zt_{n}_{multiple_tries}.pt')






if __name__ == "__main__":
    main()

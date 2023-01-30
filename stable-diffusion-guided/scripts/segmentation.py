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
    target = target[-1]
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
    operation.operation_func = None
    operation.other_guidance_func = operation_func

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = mse_loss
    operation.other_criterion = CrossEntropyLoss

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff

    operation.guidance_3 = args.optim_guidance_3
    operation.optim_guidance_3_wt = args.optim_guidance_3_wt
    operation.guidance_2 = args.optim_guidance_2
    operation.original_guidance = args.optim_original_guidance

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

def get_images(val1, n_samples):
    take_labels = [i for i in range(153,260)]

    dog_images = []
    dog_labels = []

    for batch_ind, batch in enumerate(val1):
        image, label = batch
        for i in range(label.shape[0]):
            if label[i] in take_labels:
                dog_images.append(image[i:i+1])
                dog_labels.append(label[i:i+1])

                if len(dog_images) == n_samples:
                    return dog_images, dog_labels


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
    parser.add_argument("--batch_size", default=0, type=int)



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
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1



    operation = get_optimation_details(opt)
    torch.set_grad_enabled(False)

    prompt = get_seg_text(opt.text_type)
    print(prompt)

    print('loading the dataset...')
    train_dataset, val_dataset = get_train_val_datasets(batch_size)
    print('done')
    print('splitting the dataset...')
    base = '/cmlscratch/bansal01/fall_2022/Guided_Diffusion_Imagenet/'
    train1, train2 = get_splitted_dataset(dataset=train_dataset,
                                          checkpoint_path=base + 'checkpoints/non_equal_split/partitions_train.pt')
    val1, val2 = get_splitted_dataset(dataset=val_dataset,
                                      checkpoint_path=base + 'checkpoints/non_equal_split/partitions_val.pt')
    print('done')
    train1, train2 = get_loader_from_dataset(batch_size, train1, True), get_loader_from_dataset(batch_size, train2, False)
    val1, val2 = get_loader_from_dataset(batch_size, val1, True), get_loader_from_dataset(batch_size, val2, False)

    dog_images, dog_labels = get_images(val1, opt.n_iter * batch_size)
    dog_images = torch.concat(dog_images, dim=0)
    dog_labels = torch.concat(dog_labels, dim=0)

    with open(base + 'imagenet1000_clsidx_to_labels.txt', 'r') as inf:
        dict_from_file = eval(inf.read())

    for n in trange(opt.n_iter, desc="Sampling"):

        if n==5:
            start = n * batch_size
            end = n * batch_size + batch_size

            print(start, end)

            image, label = dog_images[start: end], dog_labels[start: end]
            image, label = image.cuda(), label.cuda()
            print(label)

            p1 = dict_from_file[label.cpu().numpy()[0]]
            final_prompt = p1 + prompt
            print(p1)

            with torch.no_grad():
                map = operation.other_guidance_func(image).softmax(dim=1)

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

            mask = sep_map[:, 0:1, :, :]
            mask = TF.resize(mask, (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
            image_mask = image * mask

            utils.save_image(mask, f'{results_folder}/mask_{n}.png')
            utils.save_image((image_mask + 1) * 0.5, f'{results_folder}/image_mask_{n}.png')

            uc = None
            if opt.scale != 1.0:
                uc = model.module.get_learned_conditioning(batch_size * [""])
            c = model.module.get_learned_conditioning([final_prompt])

            for multiple_tries in range(2):
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                samples_ddim, start_zt = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta,
                                                 operated_image=[image_mask, mask, map],
                                                 operation=operation)

                x_samples_ddim = model.module.decode_first_stage(samples_ddim)
                x_samples_ddim_unnorm = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                utils.save_image(x_samples_ddim_unnorm, f'{results_folder}/new_img_{n}_{multiple_tries}.png')

                with torch.no_grad():
                    new_map = operation.other_guidance_func(x_samples_ddim)

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

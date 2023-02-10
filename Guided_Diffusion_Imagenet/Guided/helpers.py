import argparse

import torch
import torch.nn as nn
import torchvision
import torch as th
import os
import torch.nn.functional as F
from functools import partial
import torch
import random

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import torch
import numpy as np

class Jitter(nn.Module):
    def __init__(self, lim: int = 32):
        super().__init__()
        self.lim = lim

    def forward(self, x: torch.tensor) -> torch.tensor:
        off1 = random.randint(-self.lim, self.lim)
        off2 = random.randint(-self.lim, self.lim)
        return torch.roll(x, shifts=(off1, off2), dims=(2, 3))

class ColorJitter(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, mean: float = 1., std: float = 1.):
        super().__init__()
        self.batch_size, self.mean_p, self.std_p = batch_size, mean, std
        self.mean = self.std = None
        self.shuffle()
        self.shuffle_every = shuffle_every

    def shuffle(self):
        self.mean = (torch.rand((self.batch_size, 3, 1, 1,)).cuda() - 0.5) * 2 * self.mean_p
        self.std = ((torch.rand((self.batch_size, 3, 1, 1,)).cuda() - 0.5) * 2 * self.std_p).exp()

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return (img - self.mean) / self.std


class ColorJitterR(ColorJitter):
    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return (img * self.std) + self.mean


def get_model_diffusion_classifier(args):
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print('loading diffusion model...')
    print(args.model_path)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)
    print('done')
    model.eval()
    model.requires_grad_(False)
    model = torch.nn.DataParallel(model).cuda()
    if len(args.classifier_path) > 2:
        print("loading classifier...")
        classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        classifier_checkpoint = torch.load(args.classifier_path)
        classifier.load_state_dict(classifier_checkpoint)
        print('done')
        classifier.eval()
        classifier.requires_grad_(False)
        classifier = torch.nn.DataParallel(classifier).cuda()
    else:
        classifier = None

    return model, diffusion, classifier


def get_parser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_time', type=int, default=10)
    parser.add_argument('--data_dir', default='/fs/cml-datasets/ImageNet/ILSVRC2012/train')
    parser.add_argument('--val_data_dir', default='/fs/cml-datasets/ImageNet/ILSVRC2012/val')
    parser.add_argument('--workers', default=5, type=int)
    add_dict_to_argparser(parser, defaults)
    return parser


class IdentityBN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityBN, self).__init__()

    def forward(self, x):
        return x


# We may want to use a pretrained model
def get_model(in_channels=6, remove_bn=False):
    norm_layer = IdentityBN if remove_bn else nn.BatchNorm2d
    resnet18 = torchvision.models.resnet18(num_classes=2, norm_layer=norm_layer)
    resnet18.conv1 = nn.Conv2d(in_channels, resnet18.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    return resnet18


def cond_fn(x, t, y=None, args=None, classifier=None):
    assert y is not None
    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale


def model_fn(x, t, y=None, args=None, model=None):
    # assert y is not None
    return model(x, t, y if args.class_cond else None)


def get_noisy_denoised_incremental(args, model, diffusion, classifier, image, label):
    image, label = image.cuda(), label.cuda()
    times = torch.ones(image.shape[0], dtype=torch.long).cuda() * (args.max_time - 1)
    perturbed = diffusion.q_sample(image, times)
    model_kwargs = {}
    model_kwargs["y"] = label
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    sample = sample_fn(
        partial(model_fn, model=model, args=args),
        (args.batch_size, 3, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        cond_fn=partial(cond_fn, classifier=classifier, args=args),
        device=torch.device('cuda'),
        progress=True,
        max_time=args.max_time,
        noise=perturbed,
    )

    return sample


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

class OptimizerDetails:
    def __init__(self):
        self.num_steps = None
        self.operation_func = None
        self.optimizer = None # handle it on string level
        self.lr = None
        self.loss_func = None
        self.max_iters = 0
        self.loss_cutoff = None
        self.lr_scheduler = None
        self.warm_start = None
        self.old_img = None
        self.fact = 0.5
        self.print = False
        self.print_every = None
        self.folder = None
        self.tv_loss = None
        self.guidance_3 = False
        self.optim_guidance_3_wt = 0
        self.other_guidance_func = None
        self.other_criterion = None
        self.original_guidance = False
        self.sampling_type = None
        self.loss_save = None


class Operation:
    def __init__(self, args, operation, shape, progressive=False):
        self.args = args
        self.model, self.diffusion, self.classifier = get_model_diffusion_classifier(self.args)
        self.progressive = progressive
        self.operation = operation
        self.shape = shape

    def operator(self, label, operated_image, max_time=None):
        max_time = max_time or self.args.max_time
        model_kwargs = {}
        model_kwargs["y"] = label

        sample_fn = self.diffusion.ddim_sample_loop_operation
        sample = sample_fn(
            partial(model_fn, model=self.model, args=self.args),
            self.shape,
            operated_image=operated_image,
            operation=self.operation,
            clip_denoised=self.args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=partial(cond_fn, classifier=self.classifier, args=self.args),
            device=torch.device('cuda'),
            progress=self.progressive
        )

        return sample



class Normalize(object):
    def __call__(self, sample):
        return (sample - 0.5) * 2


def un_normalize(image):
    return (image + 1) / 2

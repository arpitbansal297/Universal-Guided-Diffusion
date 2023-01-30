import argparse
import os

import numpy as np
import torch
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import transforms, utils

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# PYTHONPATH=. python scripts/load_model.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=4,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


args = create_argparser().parse_args()

model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)

logger.log("loading diffusion model...")
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint)
model.eval()
model.requires_grad_(False)
model = torch.nn.DataParallel(model).cuda()
logger.log("loading classifier...")
classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
classifier_checkpoint = torch.load(args.classifier_path)
classifier.load_state_dict(classifier_checkpoint)
classifier.eval()
classifier.requires_grad_(False)
classifier = torch.nn.DataParallel(classifier).cuda()

def cond_fn(x, t, y=None):
    assert y is not None
    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale


def model_fn(x, t, y=None):
    assert y is not None
    return model(x, t, y if args.class_cond else None)


logger.log("sampling...")
all_images = []
all_labels = []
while len(all_images) * args.batch_size < args.num_samples:
    model_kwargs = {}
    classes = th.randint(
        low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
    )
    model_kwargs["y"] = classes
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    sample = sample_fn(
        model_fn,
        (args.batch_size, 3, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        device=dist_util.dev(),
        progress=True,
        start_factor=1
    )
    utils.save_image((sample+1)*0.5, 'test_2.png', nrow=6)
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)


    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    print(sample.shape)

    gathered_samples = [sample]
    all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
    gathered_labels = [classes]
    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
    logger.log(f"created {len(all_images) * args.batch_size} samples")


arr = np.concatenate(all_images, axis=0)
arr = arr[: args.num_samples]
print(arr.shape)
label_arr = np.concatenate(all_labels, axis=0)
label_arr = label_arr[: args.num_samples]
shape_str = "x".join([str(x) for x in arr.shape])
out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
logger.log(f"saving to {out_path}")
np.savez(out_path, arr, label_arr)

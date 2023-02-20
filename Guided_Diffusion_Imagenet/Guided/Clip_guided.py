import sys
sys.path.append('./')

import torchvision.transforms.functional as TF
from torch import nn, einsum

import torch
import torch.utils.data
from Guided.helpers import get_parser, Operation, OptimizerDetails
import cv2
from torchvision import transforms, utils
import os
import errno
import clip



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
parser.add_argument('--optim_backward_guidance_max_iters', type=int, default=0)
parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
parser.add_argument('--optim_original_conditioning', action='store_true', default=False)
parser.add_argument("--optim_forward_guidance_wt", default=2.0, type=float)
parser.add_argument("--optim_tv_loss", default=None, type=float)
parser.add_argument('--optim_warm_start', action='store_true', default=False)
parser.add_argument('--optim_print', action='store_true', default=False)
parser.add_argument('--optim_folder', default='./temp/')
parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
parser.add_argument("--text", default="van gogh style", type=str)
parser.add_argument("--trials", default=10, type=int)
parser.add_argument("--samples_per_diffusion", default=4, type=int)


args = parser.parse_args()

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# PYTHONPATH=. python scripts/load_model.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS

# MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
# PYTHONPATH=. python Guided/Segmentation_mobilenet.py $MODEL_FLAGS --classifier_scale 10.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS --batch_size 4



BATCH_SIZE = args.batch_size
resolution_fact = 8

clip_model, clip_preprocess = clip.load("RN50")
clip_model = torch.nn.DataParallel(clip_model).cuda()
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False


class Clip(nn.Module):
    def __init__(self, model):
        super(Clip, self).__init__()
        self.model = model
        self.trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, y):

        x = (x + 1) * 0.5
        x = TF.resize(x, (224, 224), interpolation=TF.InterpolationMode.BILINEAR)
        x = self.trans(x)

        logits_per_image, logits_per_text = clip_model(x, y)
        return -1 * logits_per_image

l_func = Clip(clip_model)
l_func.eval()
for param in l_func.parameters():
    param.requires_grad = False


results_folder = args.optim_folder
create_folder(results_folder)

operation = OptimizerDetails()

seq = []
pre = torch.nn.Sequential(*seq)

operation.num_steps = args.optim_num_steps #[2]
operation.operation_func = None
operation.other_guidance_func = None

operation.optimizer = 'Adam'
operation.lr = args.optim_lr #0.01
operation.loss_func = l_func
operation.other_criterion = None

operation.max_iters = args.optim_backward_guidance_max_iters #00
operation.loss_cutoff = args.optim_loss_cutoff #0.00001
operation.tv_loss = args.optim_tv_loss

operation.guidance_3 = args.optim_forward_guidance #True
operation.original_guidance = args.optim_original_conditioning

operation.optim_guidance_3_wt = args.optim_forward_guidance_wt

operation.warm_start = args.optim_warm_start #False
operation.print = args.optim_print
operation.print_every = 500
operation.folder = results_folder

operator = Operation(args, operation=operation, shape=[args.samples_per_diffusion, 3, 256, 256], progressive=True)
cnt = 0

def return_cv2(img, path):
    black = [255, 255, 255]
    img = (img + 1) * 0.5
    utils.save_image(img, path, nrow=1)
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
    return img


for trial in range(args.trials + 1):
    text = []
    for b in range(args.samples_per_diffusion):
        text.append(args.text)
    print(text)
    text = clip.tokenize(text).cuda()

    print("Start")
    output = operator.operator(label=None, operated_image=text)
    utils.save_image((output + 1) * 0.5, f'{results_folder}/new_img_{trial}.png')




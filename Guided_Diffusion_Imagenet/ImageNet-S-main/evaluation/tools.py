import sys
import torch
sys.path.append("..")
from data.data import EvalDataset


def get_param(mode):
    assert mode in ['50', '300', '919'], 'invalid dataset'
    params = {
        '50':  {'num_classes': 50,
                'classes': 'classes_50',
                'names': 'ImageNetS_im50_validation.txt'},
        '300': {'num_classes': 300,
                'classes': 'classes_300',
                'names': 'ImageNetS_im300_validation.txt'},
        '919': {'num_classes': 919,
                'classes': 'classes_919',
                'names': 'ImageNetS_im919_validation.txt'},
    }

    return params[mode]


def get_loader(predict_dir, gt_dir, name_list, match=None, workers=32):
    dataset = EvalDataset(predict_dir, gt_dir, name_list, match=match)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        drop_last=False)
    return loader

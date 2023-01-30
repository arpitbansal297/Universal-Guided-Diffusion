import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import json
from tools import get_loader, get_param
from hungarian import hungarian


def matching(loader, num_classes):
    labels = []
    gts = []
    for gt, predict, _, _ in tqdm(loader):
        gt = gt.cuda()
        predict = predict.cuda()

        gt = torch.unique(gt.view(-1))
        # remove background
        gt = gt - 1
        gt = gt.tolist()
        if -1 in gt:
            gt.remove(-1)
        gts.append(gt)

        label = torch.unique(predict.view(-1))
        label = label - 1
        label = label.tolist()
        if -1 in label:
            label.remove(-1)
        labels.append(label)

    match = hungarian(gts, labels, num_classes=num_classes)
    return match


def evaludation(args):

    params = get_param(args.mode)

    val_loader = get_loader(args.predict_dir, args.gt_dir,
                            name_list=os.path.join('../data', 'names', params['names']), workers=args.workers)

    match = matching(val_loader, num_classes=params['num_classes'])
    match[0] = 0
    with open('{0}.json'.format(args.session_name), 'w') as f:
        f.write(json.dumps(match))

    return match


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-dir", default='imagenet50', type=str)
    parser.add_argument("--gt-dir", default='imagenet50', type=str)
    parser.add_argument('--workers', default=32, type=int)
    parser.add_argument('--mode', type=str, default='50', choices=['50', '300', '919'])
    parser.add_argument('--session-name', type=str, required=True)
    args = parser.parse_args()

    evaludation(args)

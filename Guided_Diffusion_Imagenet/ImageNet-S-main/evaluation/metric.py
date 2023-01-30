import torch
import math

def IoUGPU(output, target, K):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    return area_intersection, area_output, area_target


def FMeasureGPU(output, target, eps=1e-20, beta=0.3):
    target = (target > 0) * 1.0
    output = (output > 0) * 1.0

    t = torch.sum(target)
    p = torch.sum(output)
    tp = torch.sum(target * output)
    recall = tp / (t + eps)
    precision = tp / (p + eps)
    f_score = (1 + beta) * precision * recall / (beta * precision + recall + eps)

    return f_score


def IoUDifferentSizeGPU(predict, gt, K, Ts, Ps, TPs):
    s5, s25, s50, s100 = [], [], [], []
    gts = {}
    predicts = {}
    bg_gt = gt[gt == 0]
    bg_predict = predict[gt == 0]

    counts = torch.bincount(gt.view(-1).long())
    size = {}
    for c in torch.unique(gt.view(-1)):
        if c == 0:
            continue
        size[int(c)] = math.ceil(100.0 * counts[int(c)] / sum(counts))

    for k, v in size.items():
        if v <= 5:
            s5.append(k)
        elif v <= 25:
            s25.append(k)
        elif v <= 50:
            s50.append(k)
        else:
            s100.append(k)

        gts[k] = gt[gt == k]
        predicts[k] = predict[gt == k]

    # different_size
    if len(s5) > 0:
        gt5 = torch.cat([bg_gt] + [gts[k] for k in s5], dim=0)
        predict5 = torch.cat([bg_predict] + [predicts[k] for k in s5], dim=0)
        area_intersection, area_output, area_target = IoUGPU(predict5, gt5, K)

        Ts[0] += area_output
        Ps[0] += area_target
        TPs[0] += area_intersection

    if len(s25) > 0:
        gt25 = torch.cat([bg_gt] + [gts[k] for k in s25], dim=0)
        predict25 = torch.cat([bg_predict] + [predicts[k] for k in s25], dim=0)
        area_intersection, area_output, area_target = IoUGPU(predict25, gt25, K)

        Ts[1] += area_output
        Ps[1] += area_target
        TPs[1] += area_intersection

    if len(s50) > 0:
        gt50 = torch.cat([bg_gt] + [gts[k] for k in s50], dim=0)
        predict50 = torch.cat([bg_predict] + [predicts[k] for k in s50], dim=0)
        area_intersection, area_output, area_target = IoUGPU(predict50, gt50, K)

        Ts[2] += area_output
        Ps[2] += area_target
        TPs[2] += area_intersection

    if len(s100) > 0:
        gt100 = torch.cat([bg_gt] + [gts[k] for k in s100], dim=0)
        predict100 = torch.cat([bg_predict] + [predicts[k] for k in s100], dim=0)
        area_intersection, area_output, area_target = IoUGPU(predict100, gt100, K)

        Ts[3] += area_output
        Ps[3] += area_target
        TPs[3] += area_intersection


def IoUDifferentSizeGPUWithBoundary(predict, gt, boundary_predict, boundary_gt, K, Ts, Ps, TPs, BTs, BPs, BTPs):
    s5, s25, s50, s100 = [], [], [], []
    gts = {}
    predicts = {}
    boundary_gts = {}
    boundary_predicts = {}
    bg_gt = gt[gt == 0]
    bg_predict = predict[gt == 0]
    bg_boundary_gt = boundary_gt[gt == 0]
    bg_boundary_predict = boundary_predict[gt == 0]

    counts = torch.bincount(gt.view(-1).long())
    size = {}
    for c in torch.unique(gt.view(-1)):
        if c == 0:
            continue
        size[int(c)] = math.ceil(100.0 * counts[int(c)] / sum(counts))

    for k, v in size.items():
        if v <= 5:
            s5.append(k)
        elif v <= 25:
            s25.append(k)
        elif v <= 50:
            s50.append(k)
        else:
            s100.append(k)

        gts[k] = gt[gt == k]
        predicts[k] = predict[gt == k]
        boundary_gts[k] = boundary_gt[gt == k]
        boundary_predicts[k] = boundary_predict[gt == k]

    # different_size
    if len(s5) > 0:
        gt5 = torch.cat([bg_gt] + [gts[k] for k in s5], dim=0)
        predict5 = torch.cat([bg_predict] + [predicts[k] for k in s5], dim=0)
        area_intersection, area_output, area_target = IoUGPU(predict5, gt5, K)

        Ts[0] += area_output
        Ps[0] += area_target
        TPs[0] += area_intersection

        # boundary
        gt5 = torch.cat([bg_boundary_gt] + [boundary_gts[k] for k in s5], dim=0)
        predict5 = torch.cat([bg_boundary_predict] + [boundary_predicts[k] for k in s5], dim=0)
        area_intersection, area_output, area_target = IoUGPU(predict5, gt5, K + 1)

        BTs[0] += area_output[1:]
        BPs[0] += area_target[1:]
        BTPs[0] += area_intersection[1:]

    if len(s25) > 0:
        gt25 = torch.cat([bg_gt] + [gts[k] for k in s25], dim=0)
        predict25 = torch.cat([bg_predict] + [predicts[k] for k in s25], dim=0)
        area_intersection, area_output, area_target = IoUGPU(predict25, gt25, K)

        Ts[1] += area_output
        Ps[1] += area_target
        TPs[1] += area_intersection

        # boundary
        gt25 = torch.cat([bg_boundary_gt] + [boundary_gts[k] for k in s25], dim=0)
        predict25 = torch.cat([bg_boundary_predict] + [boundary_predicts[k] for k in s25], dim=0)
        area_intersection, area_output, area_target = IoUGPU(predict25, gt25, K + 1)

        BTs[1] += area_output[1:]
        BPs[1] += area_target[1:]
        BTPs[1] += area_intersection[1:]

    if len(s50) > 0:
        gt50 = torch.cat([bg_gt] + [gts[k] for k in s50], dim=0)
        predict50 = torch.cat([bg_predict] + [predicts[k] for k in s50], dim=0)
        area_intersection, area_output, area_target = IoUGPU(predict50, gt50, K)

        Ts[2] += area_output
        Ps[2] += area_target
        TPs[2] += area_intersection

        # boundary
        gt50 = torch.cat([bg_boundary_gt] + [boundary_gts[k] for k in s50], dim=0)
        predict50 = torch.cat([bg_boundary_predict] + [boundary_predicts[k] for k in s50], dim=0)
        area_intersection, area_output, area_target = IoUGPU(predict50, gt50, K + 1)

        BTs[2] += area_output[1:]
        BPs[2] += area_target[1:]
        BTPs[2] += area_intersection[1:]

    if len(s100) > 0:
        gt100 = torch.cat([bg_gt] + [gts[k] for k in s100], dim=0)
        predict100 = torch.cat([bg_predict] + [predicts[k] for k in s100], dim=0)
        area_intersection, area_output, area_target = IoUGPU(predict100, gt100, K)

        Ts[3] += area_output
        Ps[3] += area_target
        TPs[3] += area_intersection

        # boundary
        gt100 = torch.cat([bg_boundary_gt] + [boundary_gts[k] for k in s100], dim=0)
        predict100 = torch.cat([bg_boundary_predict] + [boundary_predicts[k] for k in s100], dim=0)
        area_intersection, area_output, area_target = IoUGPU(predict100, gt100, K + 1)

        BTs[3] += area_output[1:]
        BPs[3] += area_target[1:]
        BTPs[3] += area_intersection[1:]

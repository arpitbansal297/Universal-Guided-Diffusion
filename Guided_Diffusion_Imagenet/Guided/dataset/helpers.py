import torch
import random
import numpy as np
from torch.utils.data.dataset import Subset


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


'''
usage: datasets = get_splitted_dataset(dataset)
'''


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

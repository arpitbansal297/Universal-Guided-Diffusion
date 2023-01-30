import os
from collections import defaultdict

from imagenet.helpers import fix_seed, get_splitted_dataset
import argparse
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()

parser.add_argument('--random_seed', default=123, type=int)
parser.add_argument('--partitions', default=6, type=int)
parser.add_argument('--root', default='/fs/cml-datasets/ImageNet/ILSVRC2012')
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--run_command', default='PYTHONPATH=. python imagenet/split_dataset.py',
                    help='How to run the script.')
parser.add_argument('--num_classes', default=1000, type=int)
parser.add_argument('--shuffle', default=False, action='store_true')

args = parser.parse_args()

fix_seed(args.random_seed)


imagenet_dataset = torchvision.datasets.ImageNet(root=args.root)

total = 0
all_labels = []
print(f'dataset length is {len(imagenet_dataset)}')
for ind, (img, label) in enumerate(imagenet_dataset):
    all_labels.append(label)

all_labels = torch.tensor(all_labels)

labels_indices = {}

for label in range(args.num_classes):
    current_inds = (all_labels == label).nonzero(as_tuple=True)[0]
    labels_indices[label] = current_inds

partitions = defaultdict(list)

for label in range(args.num_classes):
    label_index = labels_indices[label]
    if args.shuffle:
        shuffled = label_index.permute(label_index, dim=0)
    else:
        shuffled = label_index
    chunk_size = len(label_index) // args.partitions
    for partition_ind in range(args.partitions):
        end = (partition_ind + 1) * chunk_size if partition_ind != args.partitions - 1 else None
        sl = slice(partition_ind * chunk_size, end)
        partitions[partition_ind].append(shuffled[sl])

for partition_id in range(args.partitions):
    partitions[partition_id] = torch.cat(partitions[partition_id])

checkpoint_path = 'checkpoints/'
os.makedirs(checkpoint_path, exist_ok=True)

complete_path = os.path.join(checkpoint_path, 'partitions.pt')
torch.save({
    'partitions': partitions,  # a dictionary that holds the indices of different partitions.
    'count': args.partitions,
}, complete_path)

# output = get_splitted_dataset(imagenet_dataset)

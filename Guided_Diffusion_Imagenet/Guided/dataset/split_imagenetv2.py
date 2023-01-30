import os
from collections import defaultdict

import argparse
import torchvision
import torch.utils.data
from tqdm.auto import tqdm

from Guided.dataset.helpers import fix_seed
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()

parser.add_argument('--random_seed', default=123, type=int)
parser.add_argument('--per_class', default=5, type=int)
parser.add_argument('--partitions', default=2, type=int, help='number of the partitions')
parser.add_argument('--is_val', default=False, action='store_true')
parser.add_argument('--root', default='/cmlscratch/kazemira/datasets/ImageNetv2/top_images/imagenetv2-top-images-format-val/')
parser.add_argument('--bs', default=128, type=int)
parser.add_argument('--run_command', default='PYTHONPATH=. python Guided/dataset/split_imagenetv2.py',
                    help='How to run this script.')
parser.add_argument('--num_classes', default=1000, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--workers', default=5, type=int)
args = parser.parse_args()

fix_seed(args.random_seed)

split = 'val' if args.is_val else 'train'
transforms = transforms.ToTensor()


class FakeDataset(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        path, target = self.samples[index]
        return target


imagenet_dataset = FakeDataset(root=args.root, transform=transforms)
loader = torch.utils.data.DataLoader(
    imagenet_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)
total = 0
all_labels = []
print(f'dataset length is {len(imagenet_dataset)}')
for ind, label in tqdm(enumerate(imagenet_dataset), total=len(imagenet_dataset)):
    all_labels.append(label)

all_labels = torch.tensor(all_labels)

labels_indices = {}

for label in range(args.num_classes):
    current_inds = (all_labels == label).nonzero(as_tuple=True)[0]
    labels_indices[label] = current_inds

partitions = defaultdict(list)

for label in range(args.num_classes):
    label_index = labels_indices[label]
    for partition_ind in range(args.partitions):
        start, end = partition_ind * args.per_class, (partition_ind + 1) * args.per_class
        partitions[partition_ind].append(label_index[start:end])

for partition_id in range(args.partitions):
    partitions[partition_id] = torch.cat(partitions[partition_id])

checkpoint_path = 'checkpoints/equal_split/'
os.makedirs(checkpoint_path, exist_ok=True)
complete_path = os.path.join(checkpoint_path, f'partitions_imagenetv2.pt')
torch.save({
    'partitions': partitions,  # a dictionary that holds the indices of different partitions.
    'partitions_count': args.partitions,
    'per_class': args.per_class,
    'is_val': args.is_val,
}, complete_path)

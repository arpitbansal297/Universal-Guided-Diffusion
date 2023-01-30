import math
import random

from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

class Normalize(object):
    def __call__(self, sample):
        return (sample - 0.5) * 2

def load_data(
        *,
        data_dir,
        batch_size,
        image_size,
        class_cond=False,
        deterministic=False,
        random_crop=False,
        random_flip=True,
        workers=1,
        transform=None,
        ret_dataset=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = ImageNetDatasetCustom(
        image_size,
        random_crop=random_crop,
        random_flip=random_flip,
        root=data_dir,
        transform=transform
    )
    if ret_dataset:
        return dataset

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=False
        )

    return loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def get_train_val(args):
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=True,
        workers=args.workers,
        # transform=transforms.ToTensor(),
    )
    val_data = load_data(
        data_dir=args.val_data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        # transform=transforms.ToTensor(),
    )

    return data, val_data


def get_train_val_datasets(batch_size, train_random_crop=False, train_random_flip=False):
    data = load_data(
        data_dir='/fs/cml-datasets/ImageNet/ILSVRC2012/train',
        batch_size=batch_size,
        image_size=256,
        class_cond=True,
        random_crop=train_random_crop,
        random_flip=train_random_flip,
        workers=1,
        ret_dataset=True,
    )
    val_data = load_data(
        data_dir='/fs/cml-datasets/ImageNet/ILSVRC2012/val',
        batch_size=batch_size,
        image_size=256,
        class_cond=True,
        ret_dataset=True,
    )

    return data, val_data


class ImageNetDatasetCustom(datasets.ImageFolder):
    def __init__(self, resolution, random_crop=True, random_flip=True, *args, **kwargs):
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)
        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1
        sample = np.transpose(arr, [2, 0, 1])
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def get_train_val_regular(args):
    train_dataset = datasets.ImageFolder(
        args.data_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Normalize(),
        ]))

    val_dataset = datasets.ImageFolder(
        args.val_data_dir,
        transforms.Compose([
            transforms.Resize(256),  # 292
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            Normalize(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,  # important why shuffle is true here?
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader


def get_train_val_transform(args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Normalize(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),  # 292
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        Normalize(),
    ])

    return train_transform, val_transform


def get_loader_from_dataset(batch_size, dataset, shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,  # important why shuffle is true here?
        num_workers=1, pin_memory=True, drop_last=False)

    return loader

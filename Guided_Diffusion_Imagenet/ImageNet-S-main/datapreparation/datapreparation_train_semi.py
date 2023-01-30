import os
import argparse
import shutil


def make(mode, imagenet_dir, save_dir):
    assert mode in ['50', '300', '919']
    save_dir = os.path.join(save_dir, 'ImageNetS{0}'.format(mode), 'train-semi')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    categories = os.path.join('..', 'data', 'categories', 'ImageNetS_categories_im{0}.txt'.format(mode))
    train_names = os.path.join('..', 'data', 'names', 'ImageNetSemi_im{0}_train.txt'.format(mode))
    with open(categories, 'r') as f:
        categories = f.read().splitlines()
    with open(train_names, 'r') as f:
        train_names = f.read().splitlines()

    for cate in categories:
        os.makedirs(os.path.join(save_dir, cate))

    for item in train_names:
        item = item.split(' ')[0]
        cate, name = item.split('/')
        cate_src = name.split('_')[0]
        shutil.copy(os.path.join(imagenet_dir, 'train', cate_src, name), os.path.join(save_dir, cate, name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", default='imagenet', type=str)
    parser.add_argument("--save-dir", default='imagenet50', type=str)
    parser.add_argument('--mode', type=str, default='50', choices=['50', '300', '919', 'all'])
    args = parser.parse_args()

    if args.mode == 'all':
        make('50', args.imagenet_dir, args.save_dir)
        make('300', args.imagenet_dir, args.save_dir)
        make('919', args.imagenet_dir, args.save_dir)
    else:
        make(args.mode, args.imagenet_dir, args.save_dir)



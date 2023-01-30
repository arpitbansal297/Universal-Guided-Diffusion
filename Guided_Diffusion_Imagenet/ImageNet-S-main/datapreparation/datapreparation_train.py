import os
import argparse
import shutil

merge = {'n04356056': 'n04355933',
         'n04493381': 'n02808440',
         'n03642806': 'n03832673',
         'n04008634': 'n03773504',
         'n03887697': 'n15075141'}


def make(mode, imagenet_dir, save_dir, copy=False):
    assert mode in ['50', '300', '919']
    save_full_dir = os.path.join(save_dir, 'ImageNetS{0}'.format(mode), 'train-full')
    save_dir = os.path.join(save_dir, 'ImageNetS{0}'.format(mode), 'train')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    categories = os.path.join('..', 'data', 'categories', 'ImageNetS_categories_im{0}.txt'.format(mode))
    with open(categories, 'r') as f:
        categories = f.read().splitlines()
    for cate in categories:
        if not copy:
            if cate in merge.values():
                os.makedirs(os.path.join(save_dir, cate))
                for name in os.listdir(os.path.join(imagenet_dir, 'train', cate)):
                    os.symlink(os.path.join(imagenet_dir, 'train', cate, name), os.path.join(save_dir, cate, name))
            else:
                os.symlink(os.path.join(imagenet_dir, 'train', cate), os.path.join(save_dir, cate))
        else:
            shutil.copytree(os.path.join(imagenet_dir, 'train', cate), os.path.join(save_dir, cate))

    if mode == '919':
        os.symlink(os.path.join(imagenet_dir, 'train'), os.path.join(save_full_dir))
        for src, dst in merge.items():
            assert os.path.exists(os.path.join(save_dir, dst)) and not os.path.exists(os.path.join(save_dir, src))

            if not copy:
                for name in os.listdir(os.path.join(imagenet_dir, 'train', src)):
                    os.symlink(os.path.join(imagenet_dir, 'train', src, name), os.path.join(save_dir, dst, name))
            else:
                shutil.copytree(os.path.join(imagenet_dir, 'train', src), os.path.join(save_dir, dst))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", default='imagenet', type=str)
    parser.add_argument("--save-dir", default='imagenet50', type=str)
    parser.add_argument('--mode', type=str, default='50', choices=['50', '300', '919', 'all'])
    parser.add_argument('--copy', action='store_true')
    args = parser.parse_args()

    if args.mode == 'all':
        make('50', args.imagenet_dir, args.save_dir, args.copy)
        make('300', args.imagenet_dir, args.save_dir, args.copy)
        make('919', args.imagenet_dir, args.save_dir, args.copy)
    else:
        make(args.mode, args.imagenet_dir, args.save_dir, args.copy)

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--folder',
                    default='/cmlscratch/kazemira/datasets/ImageNetv2/top_images/imagenetv2-top-images-format-val')
args = parser.parse_args()

# PYTHONPATH=. python Guided/dataset/change_v2_folder_name.py

dirs = os.listdir(args.folder)

for d in dirs:
    int_name = int(d)
    os.rename(os.path.join(args.folder, d), os.path.join(args.folder, f'{int_name:04d}'))

import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='./test_segmentation/text_type_4/')
parser.add_argument('--img_index', type=int, default=0)
parser.add_argument('--imgs_to_save', type=int, default=20)

opt = parser.parse_args()


og_img = opt.folder + f'label_{opt.img_index}.png'
og_img = cv2.imread(og_img)
og_img = cv2.resize(og_img, (512, 512))
best_imgs = [og_img]

for j in range(opt.imgs_to_save):
    path = opt.folder + f'new_img_{opt.img_index}_{j}.png'
    img = cv2.imread(path)
    best_imgs.append(img)

best_imgs = cv2.hconcat(best_imgs)
cv2.imwrite(f'{opt.folder}/all_{opt.img_index}.png', best_imgs)
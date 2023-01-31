from deepface import DeepFace
import argparse
import os
import cv2


class node:
    def __init__(self, sim, result, img):
        self.sim = sim
        self.result = result
        self.img = img

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, default='./test_face/text_type_4/')
parser.add_argument('--result_file', type=str, default='res.txt')
parser.add_argument('--img_index', type=int, default=0)
parser.add_argument('--img_saved', type=int, default=20)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--deep_face_model', type=str, default="VGG-Face")

opt = parser.parse_args()

folder = opt.folder
file_name = folder + opt.result_file
img_index = opt.img_index

if not os.path.exists(file_name):
    open(file_name, "w").close()

with open(file_name, "w") as file:
    print("Analysis for img_ind : ", img_index)
    file.write(f"Analysis for img_ind : {img_index}")
    file.write("\n")

    All = []
    for j in range(opt.img_saved):
        path_1 = folder + f'new_img_{img_index}_{j}.png'
        path_2 = folder + f'og_img_{img_index}.png'
        try:
            result = DeepFace.verify(img1_path=path_1,
                                     img2_path=path_2,
                                     model_name=opt.deep_face_model)
            All.append(node(sim=result['distance'], result=result, img=path_1))

        except:
            pass

    All = sorted(All, key=lambda x: x.sim)
    og_img = folder + f'og_img_{img_index}.png'
    og_img = cv2.imread(og_img)
    og_img = cv2.resize(og_img, (512, 512))
    best_imgs = [og_img]

    for i in range(min(len(All), opt.top_k)):
        print(All[i].img)
        print(All[i].result)

        file.write(All[i].img)
        file.write("\n")

        file.write(str(All[i].result))
        file.write("\n")

        img = cv2.imread(All[i].img)
        best_imgs.append(img)

    best_imgs = cv2.hconcat(best_imgs)
    cv2.imwrite(f'{folder}/all_{img_index}.png', best_imgs)

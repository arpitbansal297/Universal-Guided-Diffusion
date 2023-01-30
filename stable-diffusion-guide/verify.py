from deepface import DeepFace
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--expt_index', type=int, default=1)
parser.add_argument('--img_index', type=int, default=0)
opt = parser.parse_args()

result = DeepFace.verify(img1_path=f"./face_g3_fast/check_{opt.expt_index}/new_img_{opt.img_index}.png", img2_path=f"./face_g3_fast/check_{opt.expt_index}/og_img_{opt.img_index}.png",
                         model_name="VGG-Face")
print(result)



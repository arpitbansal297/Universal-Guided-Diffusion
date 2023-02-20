import argparse
import os
import cv2
import os
import errno
from helper import get_face_text

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

# arpit
# cherry = {}
# cherry['20'] = [[1], []]
# cherry['21'] = [[], [1]]
# cherry['22'] = [[], []]
# cherry['23'] = [[], []]
# cherry['24'] = [[1], [5]]
# cherry['25'] = [[], [3]]
# cherry['26'] = [[4], []]
# cherry['27'] = [[], []]
# cherry['28'] = [[], []]
# cherry['29'] = [[], []]
# cherry['30'] = [[], []]
# cherry['31'] = [[2], []]
# cherry['32'] = [[], []]
# cherry['33'] = [[], []]
# cherry['34'] = [[], []]
# cherry['35'] = [[9], [3]]
# cherry['36'] = [[6], [2]]
#
#
# folder1 = "./test_face_1/all_top_k/arpit"
# folder2 = "./test_face_2/all_top_k/arpit"
#
# save = "./all_lab_faces/arpit"
#
# for k in cherry.keys():
#     best = cherry[k]
#     prompt = get_face_text(int(k))
#
#     img_1 = f"{folder1}/all_{prompt}_{k}.png"
#     img_2 = f"{folder2}/all_{prompt}_{k}.png"
#
#     best_1, best_2 = best[0], best[1]
#
#     cnt = 0
#     if len(best_1) != 0:
#         save_f = f"{save}/{prompt}"
#         create_folder(save_f)
#
#         print(img_1)
#         img = cv2.imread(img_1)
#         # print(img)
#         for ind in best_1:
#             c = ind * 512
#             p = img[:, c: c + 512]
#             cv2.imwrite(f'{save_f}/{cnt}.png', p)
#             cnt+=1
#
#     if len(best_2) != 0:
#         save_f = f"{save}/{prompt}"
#         create_folder(save_f)
#
#         img = cv2.imread(img_2)
#         for ind in best_2:
#             c = ind * 512
#             p = img[:, c: c + 512]
#             cv2.imwrite(f'{save_f}/{cnt}.png', p)
#             cnt+=1


#tom_1
# cherry = {}
# cherry['20'] = [[2], [8]]
# cherry['21'] = [[], [2]]
# cherry['22'] = [[1, 5], [2, 4]]
# cherry['23'] = [[], [1]]
# cherry['24'] = [[3], [1]]
# cherry['25'] = [[6], [7]]
# cherry['26'] = [[2], [1]]
# cherry['27'] = [[], []]
# cherry['28'] = [[], []]
# cherry['29'] = [[], []]
# cherry['30'] = [[], []]
# cherry['31'] = [[], []]
# cherry['32'] = [[2, 8], []]
# cherry['33'] = [[], []]
# cherry['34'] = [[6], [7]]
# cherry['35'] = [[2, 4], []]
# cherry['36'] = [[2, 6], [3]]
#
#
# folder1 = "./test_face_1/all_top_k/tomg_1"
# folder2 = "./test_face_2/all_top_k/tomg_1"
#
# save = "./all_lab_faces/tomg"
#
# for k in cherry.keys():
#     best = cherry[k]
#     prompt = get_face_text(int(k))
#
#     img_1 = f"{folder1}/all_{prompt}_{k}.png"
#     img_2 = f"{folder2}/all_{prompt}_{k}.png"
#
#     best_1, best_2 = best[0], best[1]
#
#     cnt = 0
#     if len(best_1) != 0:
#         save_f = f"{save}/{prompt}"
#         create_folder(save_f)
#
#         print(img_1)
#         img = cv2.imread(img_1)
#         # print(img)
#         for ind in best_1:
#             c = ind * 512
#             p = img[:, c: c + 512]
#             cv2.imwrite(f'{save_f}/1_{cnt}.png', p)
#             cnt+=1
#
#     if len(best_2) != 0:
#         save_f = f"{save}/{prompt}"
#         create_folder(save_f)
#
#         img = cv2.imread(img_2)
#         for ind in best_2:
#             c = ind * 512
#             p = img[:, c: c + 512]
#             cv2.imwrite(f'{save_f}/1_{cnt}.png', p)
#             cnt+=1



# Jonas
# cherry = {}
# cherry['20'] = [[2, 3, 6], [1, 3, 6]]
# cherry['21'] = [[], [1]]
# cherry['22'] = [[], [3, 10]]
# cherry['23'] = [[], []]
# cherry['24'] = [[2], [1, 4]]
# cherry['25'] = [[3], [1]]
# cherry['26'] = [[1, 8], [4, 6]]
# cherry['27'] = [[], []]
# cherry['28'] = [[], []]
# cherry['29'] = [[], []]
# cherry['30'] = [[], []]
# cherry['31'] = [[3], [5]]
# cherry['32'] = [[], [8]]
# cherry['33'] = [[], []]
# cherry['34'] = [[1], [2, 4]]
# cherry['35'] = [[4], [1, 4]]
# cherry['36'] = [[1], [4, 5]]
#
#
# folder1 = "./test_face_1/all_top_k/jonas"
# folder2 = "./test_face_2/all_top_k/jonas"
#
# save = "./all_lab_faces/jonas"
#
# for k in cherry.keys():
#     best = cherry[k]
#     prompt = get_face_text(int(k))
#
#     img_1 = f"{folder1}/all_{prompt}_{k}.png"
#     img_2 = f"{folder2}/all_{prompt}_{k}.png"
#
#     best_1, best_2 = best[0], best[1]
#
#     print(prompt, k)
#
#     cnt = 0
#     if len(best_1) != 0:
#         save_f = f"{save}/{prompt}"
#         create_folder(save_f)
#
#         # print(img_1)
#         img = cv2.imread(img_1)
#         # print(img)
#         for ind in best_1:
#             c = ind * 512
#             p = img[:, c: c + 512]
#             cv2.imwrite(f'{save_f}/{cnt}.png', p)
#             cnt+=1
#
#     if len(best_2) != 0:
#         save_f = f"{save}/{prompt}"
#         create_folder(save_f)
#
#         img = cv2.imread(img_2)
#         for ind in best_2:
#             c = ind * 512
#             p = img[:, c: c + 512]
#             cv2.imwrite(f'{save_f}/{cnt}.png', p)
#             cnt+=1





#tom_1
# cherry = {}
# cherry['20'] = [[1], [4, 9]]
# cherry['21'] = [[1], []]
# cherry['22'] = [[], [1, 2]]
# cherry['23'] = [[], []]
# cherry['24'] = [[3, 7], [1, 4, 9]]
# cherry['25'] = [[8], [7, 9]]
# cherry['26'] = [[1, 3], [7]]
# cherry['27'] = [[], []]
# cherry['28'] = [[], []]
# cherry['29'] = [[], []]
# cherry['30'] = [[], []]
# cherry['31'] = [[4], [5]]
# cherry['32'] = [[7], [8]]
# cherry['33'] = [[], []]
# cherry['34'] = [[3, 9], [3]]
# cherry['35'] = [[1, 7], [4]]
# cherry['36'] = [[1, 2, 7], [2, 7]]
#
#
# folder1 = "./test_face_1/all_top_k/tomg_2"
# folder2 = "./test_face_2/all_top_k/tomg_2"
#
# save = "./all_lab_faces/tomg"
#
# for k in cherry.keys():
#     best = cherry[k]
#     prompt = get_face_text(int(k))
#
#     img_1 = f"{folder1}/all_{prompt}_{k}.png"
#     img_2 = f"{folder2}/all_{prompt}_{k}.png"
#
#     best_1, best_2 = best[0], best[1]
#
#     cnt = 0
#     if len(best_1) != 0:
#         save_f = f"{save}/{prompt}"
#         create_folder(save_f)
#
#         print(img_1)
#         img = cv2.imread(img_1)
#         # print(img)
#         for ind in best_1:
#             c = ind * 512
#             p = img[:, c: c + 512]
#             cv2.imwrite(f'{save_f}/2_{cnt}.png', p)
#             cnt+=1
#
#     if len(best_2) != 0:
#         save_f = f"{save}/{prompt}"
#         create_folder(save_f)
#
#         img = cv2.imread(img_2)
#         for ind in best_2:
#             c = ind * 512
#             p = img[:, c: c + 512]
#             cv2.imwrite(f'{save_f}/1_{cnt}.png', p)
#             cnt+=1


#micah

cherry = {}
cherry['20'] = [[], []]
cherry['21'] = [[1], [1]]
cherry['22'] = [[1], []]
cherry['23'] = [[], []]
cherry['24'] = [[2, 5], []]
cherry['25'] = [[8], [3]]
cherry['26'] = [[1, 2], [4]]
cherry['27'] = [[], []]
cherry['28'] = [[], []]
cherry['29'] = [[], []]
cherry['30'] = [[], []]
cherry['31'] = [[2, 3, 5], []]
cherry['32'] = [[], []]
cherry['33'] = [[], []]
cherry['34'] = [[2], [7, 10]]
cherry['35'] = [[4], [4]]
cherry['36'] = [[2, 3], [1]]


folder1 = "./test_face_1/all_top_k/micah"
folder2 = "./test_face_2/all_top_k/micah"

save = "./all_lab_faces/micah"

for k in cherry.keys():
    best = cherry[k]
    prompt = get_face_text(int(k))

    img_1 = f"{folder1}/all_{prompt}_{k}.png"
    img_2 = f"{folder2}/all_{prompt}_{k}.png"

    best_1, best_2 = best[0], best[1]

    cnt = 0
    if len(best_1) != 0:
        save_f = f"{save}/{prompt}"
        create_folder(save_f)

        print(img_1)
        img = cv2.imread(img_1)
        # print(img)
        for ind in best_1:
            c = ind * 512
            p = img[:, c: c + 512]
            cv2.imwrite(f'{save_f}/2_{cnt}.png', p)
            cnt+=1

    if len(best_2) != 0:
        save_f = f"{save}/{prompt}"
        create_folder(save_f)

        img = cv2.imread(img_2)
        for ind in best_2:
            c = ind * 512
            p = img[:, c: c + 512]
            cv2.imwrite(f'{save_f}/1_{cnt}.png', p)
            cnt+=1



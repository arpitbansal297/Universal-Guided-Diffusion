import os
import errno
import cv2

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

main_dir = './paper_face_2'
create_folder(main_dir)

list = ['./face_final_original_3/cherry_face_final_original_3.png',
        './face_final_original_2/cherry_face_final_original_2.png',
        './face_final_original_1/cherry_face_final_original_1.png']

cherrys = []
for img in list:
    cherry = []
    all = cv2.imread(img)
    for i in range(all.shape[1] // 512):
        cherry.append(all[:, 512*i: 512*i + 512, :])
    cherrys.append(cherry)

to_take = [0, 1, 6, 5, 7, 9]

for t in to_take:
    all = []
    for i in range(len(cherrys)):
        cv2.imwrite(f'{main_dir}/celeb_{i}_type_{t}.png', cherrys[i][t])
        all.append(cherrys[i][t])
    all = cv2.hconcat(all)
    cv2.imwrite(f'{main_dir}/type_{t}.png', all)



# for i in range(len(cherrys)):
#     all = []
#     for t in to_take:
#         all.append(cherrys[i][t])
#     all = cv2.vconcat(all)
#     cv2.imwrite(f'{main_dir}/celeb_{i}.png', all)







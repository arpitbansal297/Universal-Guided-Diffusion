from PIL import Image
import PIL
import cv2

main_folder = "./Clip_final/"
folder = "clip_wt_5_times_5_text_type_11_cl_free/"

folder = main_folder + folder
to_save = ['0_3', '1_3', '4_8', '5_3', '6_4', '9_1', '9_4', '10_4', '3_3', '1_8']

cnt = 0
for s in to_save:
    info = s.split("_")
    img_num = int(info[0])
    img_ind = int(info[1]) - 1
    print(img_num, img_ind)

    all_img = folder + f'new_img_{img_num}.png'
    all_img = cv2.imread(all_img)
    start = 2 + img_ind * 258
    end = start + 256

    img = all_img[2:258, start:end]
    print(img.shape)

    cv2.imwrite(f'{folder}/final_{cnt}.png', img)
    cnt += 1



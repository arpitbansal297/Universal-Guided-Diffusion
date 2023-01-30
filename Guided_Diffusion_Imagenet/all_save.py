from PIL import Image
import PIL

main_folder = "./Clip_12_8/"
folder_lists = []
save_lists = []

for i in range(11, 12):
    folder_lists.append(f"clip_wt_2_times_5_text_type_{i}_ung")
    folder_lists.append(f"clip_wt_3_times_5_text_type_{i}_ung")
    folder_lists.append(f"clip_wt_2_times_10_text_type_{i}_ung")
    folder_lists.append(f"clip_wt_3_times_10_text_type_{i}_ung")

    save_lists.append(f"{i}_wt_2_times_5_ung")
    save_lists.append(f"{i}_wt_3_times_5_ung")
    save_lists.append(f"{i}_wt_2_times_10_ung")
    save_lists.append(f"{i}_wt_3_times_10_ung")


    folder_lists.append(f"clip_wt_2_times_5_text_type_{i}")
    folder_lists.append(f"clip_wt_3_times_5_text_type_{i}")
    folder_lists.append(f"clip_wt_2_times_10_text_type_{i}")
    folder_lists.append(f"clip_wt_3_times_10_text_type_{i}")

    save_lists.append(f"{i}_wt_2_times_5")
    save_lists.append(f"{i}_wt_3_times_5")
    save_lists.append(f"{i}_wt_2_times_10")
    save_lists.append(f"{i}_wt_3_times_10")



    folder_lists.append(f"clip_wt_2_times_5_text_type_{i}_cl_free")
    folder_lists.append(f"clip_wt_3_times_5_text_type_{i}_cl_free")
    folder_lists.append(f"clip_wt_2_times_10_text_type_{i}_cl_free")
    folder_lists.append(f"clip_wt_3_times_10_text_type_{i}_cl_free")

    save_lists.append(f"{i}_wt_2_times_5_cl_free")
    save_lists.append(f"{i}_wt_3_times_5_cl_free")
    save_lists.append(f"{i}_wt_2_times_10_cl_free")
    save_lists.append(f"{i}_wt_3_times_10_cl_free")



for i in range(len(folder_lists)):
    f = folder_lists[i]
    s = save_lists[i]

    img_file = main_folder + f + "/new_img_0.png"
    im1 = Image.open(img_file)
    im1.save(main_folder + "all_final_2/" + s + ".png")

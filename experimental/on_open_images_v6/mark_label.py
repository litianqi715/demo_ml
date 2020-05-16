import pandas as pd
import numpy as np
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import copy
import os
from time import time
import json

label_map_path = "D:\\open_images\\4metadata\\oidv6-class-descriptions.csv"
train_label_path = "D:\\open_images\\1human\\oidv6-train-annotations-human-imagelabels.csv"
train_label_json = "D:\\open_images\\1human\\oidv6-train-annotations-human-imagelabels.json"
val_label_path = "D:\\open_images\\1human\\validation-annotations-human-imagelabels-boxable.csv"
test_label_path = "D:\\open_images\\1human\\test-annotations-human-imagelabels.csv"

raw_img_base = "D:\\open_images\\raw_image\\"
check_img_base = "D:\\open_images\\check_image\\"
raw_label_base = ""
demo_path = "C:\\Users\\Administrator\\Desktop\\"


start_time = time()

def print_time(msg):
    global start_time 
    print("{}--- {}".format(time()-start_time, msg))
    start_time = time()


def mark_label_main():
    print_time("start check label")
    # get label_id_to_name dict
    df = pd.read_csv(label_map_path)
    label_ids = np.array(df['LabelName'])
    label_names = np.array(df['DisplayName'])
    label_id_to_name = dict()
    for idx, (label_id, label_name) in enumerate(zip(label_ids, label_names)):
        label_id_to_name[label_id] = label_name
    print_time("got label map dict")

    # get train_imgs_label dict
    train_imgs_label = json.load(open(train_label_json))
    # df = pd.read_csv(train_label_path)
    # image_ids = np.array(df['ImageID'])
    # label_names = np.array(df['LabelName'])
    # confidences = np.array(df['Confidence']) > 0
    # valid_image_ids = image_ids[confidences]
    # valid_label_ids = label_names[confidences]
    
    # train_imgs_label = defaultdict(list)
    # for idx, (image_id, label_id) in enumerate(zip(valid_image_ids, valid_label_ids)):
    #     train_imgs_label[image_id].append( label_id_to_name[label_id] )
    # del image_ids, label_names, confidences, valid_image_ids, valid_label_ids
    print_time("got img label list")

    # manual check train00
    for folder_name in ["train_00"]:
        mark_one_folder(folder_name, train_imgs_label)


def mark_one_folder(folder_name, train_imgs_label):
    print_time("start processing %s" % (folder_name))
    folder_path = raw_img_base + folder_name + "\\"
    check_path =  check_img_base + folder_name + "\\"
    if not os.path.exists(check_path):
        os.makedirs(check_path)
        print_time("create folder {}".format(check_path))
    file_cnt = 0
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if not filename.endswith(".jpg"):
                continue
            fileid = filename.split(".")[0]
            out_file_name = os.path.join(check_path, filename)
            file_cnt+=1
            if file_cnt % 5000 == 0:
                print_time("get 5000 images")
            try:
                mark_one_file(os.path.join(root, filename), fileid, out_file_name, train_imgs_label)
            except Exception as ex:
                print(ex)
                import pdb
                pdb.set_trace()
                pass
    print("end one folder")


def mark_one_file(filename, fileid, out_file_name, train_imgs_label):
    demo_img_path = filename
    demo_img_id = fileid

    demo_img = cv2.imread(demo_img_path)
    demo_img_labels = train_imgs_label[demo_img_id]
    demo_img_shape = demo_img.shape  # y,x,c
    # print(demo_img_labels)

    font = cv2.FONT_HERSHEY_SIMPLEX
    demo_img_text = copy.deepcopy(demo_img)
    for idx, one_label in enumerate(demo_img_labels):
        posy = 50 + idx * 30
        _ = cv2.putText(demo_img_text, one_label, (50, posy), font, 1, (255, 255, 255), 2)
    cv2.imwrite(out_file_name, demo_img_text)
    # plt.title("demo")
    # plt.imshow(demo_img)
    # plt.show()
    # print("--------")
    pass


if __name__ == "__main__":
    check_label()

import pandas as pd
import numpy as np
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import copy


label_map_path = "D:\\project_open_images\\4metadata\\oidv6-class-descriptions.csv"
train_label_path = "D:\\project_open_images\\1human\\oidv6-train-annotations-human-imagelabels.csv"
val_label_path = "D:\\project_open_images\\1human\\validation-annotations-human-imagelabels-boxable.csv"
test_label_path = "D:\\project_open_images\\1human\\test-annotations-human-imagelabels.csv"
raw_img_base = "D:\\project_open_images\\raw_image\\"
raw_label_base = ""
demo_path = "C:\\Users\\Administrator\\Desktop\\"

if __name__ == "__main__":
    visual_img_label()

def visual_img_label():
    # label id to name dict
    df = pd.read_csv(label_map_path)
    label_ids = np.array(df['LabelName'])
    label_names = np.array(df['DisplayName'])
    label_id_to_name = dict()
    for idx, (label_id, label_name) in enumerate(zip(label_ids, label_names)):
        label_id_to_name[label_id] = label_name
    print("--- got label map dict")

    # get labels of img dict
    df = pd.read_csv(train_label_path)
    image_ids = np.array(df['ImageID'])
    label_names = np.array(df['LabelName'])
    confidences = np.array(df['Confidence']) > 0
    valid_image_ids = image_ids[confidences]
    valid_label_ids = label_names[confidences]
    train_imgs_label = defaultdict(list)
    for idx, (image_id, label_id) in enumerate(zip(valid_image_ids, valid_label_ids)):
        train_imgs_label[image_id].append( label_id_to_name[label_id] )
    print("--- got img label list")

    # manual check train00
    train00_raw_img_path = raw_img_base + "train_00\\"
    

def mark_one_folder(folder_name):
    demo_img_path = train00_raw_img_path + "dec2aafe511bcd90.jpg"
    demo_img = cv2.imread(demo_img_path)
    demo_img_id = "dec2aafe511bcd90.jpg".split(".")[0]
    demo_img_labels = ['Water', 'Plant', 'Tree']  # train_imgs_label[demo_img_id]
    demo_img_shape = demo_img.shape  # y,x,c
    print(demo_img_labels)

    font = cv2.FONT_HERSHEY_SIMPLEX
    demo_img_text = copy.deepcopy(demo_img)
    for idx, one_label in enumerate(demo_img_labels):
        posy = 50 + idx * 30
        _ = cv2.putText(demo_img_text, one_label, (50, posy), font, 1, (255, 255, 255), 2)
    cv2.imwrite(demo_path+"dec2aafe511bcd90_label.jpg", demo_img_text)
    # plt.title("demo")
    # plt.imshow(demo_img)
    # plt.show()
    print("--------")
    pass

# stat label dist
def get_label_stat():
    df = pd.read_csv(val_label_path)
    image_ids = np.array(df['ImageID'])
    label_names = np.array(df['LabelName'])
    confidences = np.array(df['Confidence']) > 0

    valid_image_ids = image_ids[confidences]
    valid_label_names = label_names[confidences]

    class_name_occurs = defaultdict(int)
    image_contain_labels = defaultdict(int)

    for idx, (image_id, label_name) in enumerate(zip(valid_image_ids, valid_label_names)):
        class_name_occurs[label_name] += 1
        image_contain_labels[image_id] += 1

    labels_occurrence_time = list(map(int, class_name_occurs.values()))
    labels_in_image = list(map(int, image_contain_labels.values()))

    print("lines: %d, valid lines: %d\n" % (len(confidences), len(valid_image_ids)))
    print("No. of labels occured: {}\nlabels average occurrence: {}\n"
        .format(len(class_name_occurs.keys()), np.mean(labels_occurrence_time)))
    print("image average labels: {}\n".format(np.mean(labels_in_image)))

    print("done")
    #print(df)

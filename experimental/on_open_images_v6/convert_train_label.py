import time
import numpy as np
import pandas as pd
import csv
from collections import defaultdict
import json
import os
import xlrd

label_map_path = "D:\\open_images\\4metadata\\oidv6-class-descriptions.csv"

train_label_path = "D:\\open_images\\1human\\oidv6-train-annotations-human-imagelabels.csv"
# train_json_file = "D:\\open_images\\1human\\oidv6-train-annotations-human-imagelabels.json"
val_label_path = "D:\\open_images\\1human\\validation-annotations-human-imagelabels.csv"
# val_json_file = "D:\\open_images\\1human\\validation-annotations-human-imagelabels.json"
test_label_path = "D:\\open_images\\1human\\test-annotations-human-imagelabels.csv"
# test_json_file = "D:\\open_images\\1human\\test-annotations-human-imagelabels.json"

json_hierarchy_file = "D:\\open_images\\bbox_labels_600_hierarchy.json"


# get label_id_to_name dict
df = pd.read_csv(label_map_path)
label_ids = np.array(df['LabelName'])
label_names = np.array(df['DisplayName'])
label_id_to_name = dict()
for idx, (label_id, label_name) in enumerate(zip(label_ids, label_names)):
    label_id_to_name[label_id] = label_name
print("got label map dict")


def convert_json_hierarchy(hierarchy):
    if isinstance(hierarchy, dict):
        for key, val in hierarchy.items():
            if key == "LabelName":
                if val in label_id_to_name:
                    label_name = label_id_to_name[val]
                    hierarchy[key] = label_name
            elif isinstance(val, list):
                    hierarchy[key] = convert_json_hierarchy(val)
    elif isinstance(hierarchy, list):
        for idx, elem in enumerate(hierarchy):
            hierarchy[idx] = convert_json_hierarchy(elem)
    return hierarchy
# stat label hierarchy file
# json_hierarchy = json.load(open(json_hierarchy_file))
# json_hierarchy_name = convert_json_hierarchy(json_hierarchy)
# json.dump(json_hierarchy_name, open("D:\\open_images\\bbox_labels_600_hierarchy.name.json", "w"))


def convert_label_to_json(label_path, use_cls_map=None):
    # convert cls gt labels from csv to json
    print("[+] processing {}".format(label_path))

    # get imgs_label dict
    start_t = time.time()
    df = pd.read_csv(label_path)
    image_ids = np.array(df['ImageID'])
    label_names = np.array(df['LabelName'])
    confidences = np.array(df['Confidence']) > 0
    valid_image_ids = image_ids[confidences]
    valid_label_ids = label_names[confidences]

    imgs_label = defaultdict(list)
    if use_cls_map is not None:
        imgs_cls = defaultdict(lambda : [0]*135)
        wb = xlrd.open_workbook(use_cls_map)
        names_sheet = wb.sheets()[1]
        cls_list = names_sheet.col_values(2)
        cls_list.insert(0, "any")
        print("[+] loaded cls map list")

    for idx, (image_id, label_id) in enumerate(zip(valid_image_ids, valid_label_ids)):
        imgs_label[image_id].append( label_id_to_name[label_id] )
        try:
            mapped_cls_idx = cls_list.index(label_id)
            imgs_cls[image_id][mapped_cls_idx] = 1
            imgs_cls[image_id][0] = 1
        except:
            pass
    del image_ids, label_names, confidences, valid_image_ids, valid_label_ids
    print("got img label list, use time {}".format(time.time() - start_t))
    
    json_file = os.path.splitext(label_path)[0] + ".json"
    cls_json_file = os.path.splitext(label_path)[0] + "_cls.json"
    json.dump(imgs_label, open(json_file, "w"))
    json.dump(imgs_cls, open(cls_json_file, "w"))
    del imgs_label, imgs_cls
    print("saved")

    start_t = time.time()
    imgs_label = json.load(open(json_file))
    print("reload img label list, use time {}".format(time.time() - start_t))

convert_label_to_json(test_label_path, use_cls_map="D:\\open_images\\1human\\test-annotations-human-imagelabels.csv.stat.xlsx")
convert_label_to_json(val_label_path, use_cls_map="D:\\open_images\\1human\\test-annotations-human-imagelabels.csv.stat.xlsx")
convert_label_to_json(train_label_path, use_cls_map="D:\\open_images\\1human\\test-annotations-human-imagelabels.csv.stat.xlsx")



print("done")

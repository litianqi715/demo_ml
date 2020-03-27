import time
import numpy as np
import pandas as pd
import csv
from collections import defaultdict
import json

label_map_path = "D:\\open_images\\4metadata\\oidv6-class-descriptions.csv"

train_label_path = "D:\\open_images\\1human\\oidv6-train-annotations-human-imagelabels.csv"
train_json_file = "D:\\open_images\\1human\\oidv6-train-annotations-human-imagelabels.json"

val_label_path = "D:\\open_images\\1human\\validation-annotations-human-imagelabels.csv"
val_json_file = "D:\\open_images\\1human\\validation-annotations-human-imagelabels.json"

test_label_path = "D:\\open_images\\1human\\test-annotations-human-imagelabels.csv"
test_json_file = "D:\\open_images\\1human\\test-annotations-human-imagelabels.json"

json_hierarchy_file = "D:\\open_images\\bbox_labels_600_hierarchy.json"


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
        
    for idx, (image_id, label_id) in enumerate(zip(valid_image_ids, valid_label_ids)):
        imgs_label[image_id].append( label_id_to_name[label_id] )
    del image_ids, label_names, confidences, valid_image_ids, valid_label_ids
    print("got img label list, use time {}".format(time.time() - start_t))
    json.dump(imgs_label, open(json_file, "w"))
    del imgs_label
    print("saved")

    start_t = time.time()
    imgs_label = json.load(open(json_file))
    print("reload img label list, use time {}".format(time.time() - start_t))




# get label_id_to_name dict
df = pd.read_csv(label_map_path)
label_ids = np.array(df['LabelName'])
label_names = np.array(df['DisplayName'])
label_id_to_name = dict()
for idx, (label_id, label_name) in enumerate(zip(label_ids, label_names)):
    label_id_to_name[label_id] = label_name
print("got label map dict")




print("done")

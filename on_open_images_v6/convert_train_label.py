import time
import numpy as np
import pandas as pd
from collections import defaultdict
import json

label_map_path = "D:\\open_images\\4metadata\\oidv6-class-descriptions.csv"
train_label_path = "D:\\open_images\\1human\\oidv6-train-annotations-human-imagelabels.csv"
json_file = "D:\\open_images\\1human\\oidv6-train-annotations-human-imagelabels.json"


 # get label_id_to_name dict
df = pd.read_csv(label_map_path)
label_ids = np.array(df['LabelName'])
label_names = np.array(df['DisplayName'])
label_id_to_name = dict()
for idx, (label_id, label_name) in enumerate(zip(label_ids, label_names)):
    label_id_to_name[label_id] = label_name
print("got label map dict")

# get train_imgs_label dict
start_t = time.time()
df = pd.read_csv(train_label_path)
image_ids = np.array(df['ImageID'])
label_names = np.array(df['LabelName'])
confidences = np.array(df['Confidence']) > 0
valid_image_ids = image_ids[confidences]
valid_label_ids = label_names[confidences]

train_imgs_label = defaultdict(list)
for idx, (image_id, label_id) in enumerate(zip(valid_image_ids, valid_label_ids)):
    train_imgs_label[image_id].append( label_id_to_name[label_id] )
del image_ids, label_names, confidences, valid_image_ids, valid_label_ids
print("got img label list, use time {}".format(time.time() - start_t))
json.dump(train_imgs_label, open(json_file, "w"))
del train_imgs_label
print("saved")

start_t = time.time()
train_imgs_label = json.load(open(json_file))
print("reload img label list, use time {}".format(time.time() - start_t))


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
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


class StatLabel():
    def __init__(self):
        self.train_class_name_occurs = None
        self.val_class_name_occurs = None
        self.test_class_name_occurs = None

    def get_label_stat(self, label_path, label_type, save_cnt=False, print_stat=True):
        # stat cls gt labels
        print("[+] stat label file {}".format(label_path) )

        df = pd.read_csv(label_path)
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
        if label_type == "train":
            self.train_class_name_occurs = class_name_occurs
        elif label_type == "val":
            self.val_class_name_occurs = class_name_occurs
        elif label_type == "test":
            self.test_class_name_occurs = class_name_occurs

        labels_occurrence_time = list(map(int, class_name_occurs.values()))
        labels_in_image = list(map(int, image_contain_labels.values()))

        # save label occurences
        if save_cnt:
            save_file = label_path + ".stat.tsv"
            out_writer = csv.writer(open(save_file, "w", newline=''), dialect="excel")
            out_writer.writerow(["label_name\tlabel_id\ttest\tval\ttrain"])
            for label_id, label_occurences in \
                sorted(class_name_occurs.items(), key=lambda kv:(kv[1], kv[0]), reverse=True):
                label_name = str(label_id_to_name[label_id].encode("utf-8"))
                in_train_cnt = self.train_class_name_occurs[label_id]
                in_val_cnt = self.val_class_name_occurs[label_id]
                out_writer.writerow([
                    "{}\t{}\t{}\t{}\t{}".format(label_name, label_id, label_occurences, in_val_cnt, in_train_cnt)])

        if print_stat:
            print("total imgs: %d\n" % (len(image_contain_labels.keys())) )
            print("total labels: %d, valid labels: %d\n" % (len(confidences), len(valid_image_ids)))
            print("No. of labels occured: {}\nlabels average occurrence: {}\n"
                .format(len(class_name_occurs.keys()), np.mean(labels_occurrence_time)))
            print("image average labels: {}\n".format(np.mean(labels_in_image)))
        
        return 1
    
    def demo_run(self):
        stat_label = StatLabel()
        stat_label.get_label_stat(train_label_path, label_type="train", save_cnt=False)
        stat_label.get_label_stat(val_label_path, label_type="val", save_cnt=False)
        stat_label.get_label_stat(test_label_path, label_type="test", save_cnt=True)


# get label_id_to_name dict
df = pd.read_csv(label_map_path)
label_ids = np.array(df['LabelName'])
label_names = np.array(df['DisplayName'])
label_id_to_name = dict()
for idx, (label_id, label_name) in enumerate(zip(label_ids, label_names)):
    label_id_to_name[label_id] = label_name
print("got label map dict")
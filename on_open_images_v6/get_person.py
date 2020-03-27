import time
import numpy as np
import pandas as pd
from collections import defaultdict
import json


person_category_list = ["Person", "Man", "Woman", "Boy", "Girl"]
all_category_list = person_category_list + "None"


train_label_json = "D:\\open_images\\1human\\oidv6-train-annotations-human-imagelabels.json"



# get label_id_to_name dict
df = pd.read_csv(label_map_path)
label_ids = np.array(df['LabelName'])
label_names = np.array(df['DisplayName'])
label_id_to_name = dict()
for idx, (label_id, label_name) in enumerate(zip(label_ids, label_names)):
    label_id_to_name[label_id] = label_name
print("got label map dict")



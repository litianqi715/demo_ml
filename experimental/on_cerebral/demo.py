import sys
import os
import json
sys.path.insert(0, ".\Kfbreader-win10-python36")
import kfbReader as  kr
print("good to run")
import cv2



def  get_roi(label):
	with  open(label,"r") as f:
		js = json.load(f)
	rois = []
	roi = {}
	for dic in js:
		if dic["class"] ==  "roi":
			roi = dic
			roi["poses"] = []
			rois.append(roi)
		else :
			pass
	for dic in js:
		if dic["class"] ==  "roi":
			pass
		else:
			for roi1 in rois:
				if roi1["x"] <= dic["x"] and roi1["y"] <= dic["y"] and dic["x"] + dic["w"] <= roi1["x"] + roi1["w"] and dic["y"] + dic["h"] <= roi1["y"] + roi1["h"]:
					roi1["poses"].append(dic)
	return rois

# some paths
data_path_pos = "..\data\pos"
label_path = "..\labels"
tmp_path = "C:\\tmp_cervical"
pos_files = os.listdir(data_path_pos)
save_dir = "."

# some hyper parameters
scale = 20

for ii, name1 in enumerate(pos_files):
    # some path for one image
    pos_file = os.path.join(data_path_pos, name1)
    pos_file_name = name1.split(".")[0]
    json_file = os.path.join(label_path, pos_file_name+".json")
    # save images
    reader = kr.reader()
    reader.ReadInfo(pos_file, scale, False)
    width = reader.getWidth()
    height = reader.getHeight()
    print("processing", pos_file, width, height, json_file)
    rois = get_roi(json_file)
    for idx, roi1 in  enumerate(rois):
        roi = reader.ReadRoi(roi1["x"],roi1["y"],roi1["w"],roi1["h"], scale)
        for pos in roi1["poses"]:
            rx = pos["x"]-roi1["x"]
            ry = pos["y"]-roi1["y"]
            cv2.rectangle(roi, (rx,ry), (rx+pos["w"],ry+pos["h"]),(0,255,0), 4)
        save_name = os.path.join(tmp_path, pos_file_name+"_roi"+str(idx)+".jpg")  
        cv2.imwrite(save_name,roi)
        print("save roi img:"+save_name)
    if ii % 10 == 0:
        print("processed %d imgs" % ii)


import cv2
import numpy as np

train_imgs = [
    "buildings_train_20200107\Subset1_img_RGB.tif",
    "buildings_train_20200107\Subset2_img_RGB.tif",
    "buildings_train_20200107\Subset3_img_RGB.tif",
    "buildings_train_20200107\Subset4_img_RGB.tif",
    "buildings_train_20200107\Subset5_img_RGB.tif",
]
train_anns = [
    "buildings_train_20200107\Subset1_building_footprints.tif",
    "buildings_train_20200107\Subset2_building_footprints.tif",
    "buildings_train_20200107\Subset3_building_footprints.tif",
    "buildings_train_20200107\Subset4_building_footprints.tif",
    "buildings_train_20200107\Subset5_building_footprints.tif",
]

val_imgs = ["buildings_train_20200107\Subset6_img_RGB.tif"]

test_imgs = [
    "buildings_train_20200107\Subset7_img_RGB.tif",
    "buildings_train_20200107\Subset8_img_RGB.tif",
]


img = cv2.imread("raw\Subset1_img_RGB.tif", -1) # yxc
ann = cv2.imread("raw\Subset1_building_footprints.tif", -1)
print(img.shape, ann.shape)
ann1 = np.expand_dims(ann, axis=2)
ann3 = np.concatenate((ann1, ann1, ann1), axis=-1)
ann_img = img * ann3
raw_x, raw_y = ann_img.shape[0:2]

x1, y1 = 0, 0
while x1 < raw_x - 224:
    x2 = x1 + 224
    x2 = raw_x if x2 >= raw_x or x2
    while y1 < raw_y - 224:
        y2 = y1 + 224
        y2 = raw_y if y2 >= raw_y or y2
        patch_raw = img[x1:x2, y1:y2]
        patch_ann = ann[x1:x2, y1:y2]
        patch_visual = ann_img[x1:x2, y1:y2]

print("ann image", ann_img.shape)



# for img_path in train_imgs:
#     img = cv2.imread("buildings_train_20200107\Subset1_img_RGB.tif", -1)
#     print(img_path, img.shape)

### quick visual whole image Subset1
# img = cv2.imread("buildings_train_20200107\Subset1_img_RGB.tif", -1) # yxc
# ann = cv2.imread("buildings_train_20200107\Subset1_building_footprints.tif", -1)
# print(img.shape, ann.shape)
# ann1 = np.expand_dims(ann, axis=2)
# ann3 = np.concatenate((ann1, ann1, ann1), axis=-1)
# ann_img = img * ann3
# print("ann image", ann_img.shape)
# cv2.namedWindow("Image")
# demo1 = ann_img[0:1080, 961:1920, 0]
# demo2 = ann_img[0:1080, 961:1920, 1]
# demo3 = ann_img[0:1080, 961:1920, 2]
# hmerge = np.hstack((demo1, demo2, demo3)) #水平拼接
# cv2.imshow("Image", ann_img[4000:5126, 1000:3876, :])
# cv2.waitKey(0)
# cv2.destroyAllWindows() 

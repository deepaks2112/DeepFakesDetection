from timm.models.efficientnet import efficientnet_b0
from classifier import DeepFakeClassifier
import numpy as np
import os
from torchvision.transforms import Resize
# from preprocessing import Preprocessor
import matplotlib.pyplot as plt
import cv2
from augmentations import blackout_convex_hull, remove_mouth, remove_eyes, remove_nose, remove_landmark
from augmentations import detector, predictor, blend_original
from preprocessing import extract_ori_crops_and_bboxes, extract_fake_crops, read_frames_df, extract_diff_masks
from preprocessing import extract_landmarks
import pandas as pd
# from preprocessing import frames_df
from time import time
import json
import matplotlib.pyplot as plt

fake_path = '/home/deepak/DEEPFAKES_IITBHILAI/dfdc/dfdc_train_part_01/dfdc_train_part_1/bfeewgzrbr.mp4'
real_path = '/home/deepak/DEEPFAKES_IITBHILAI/dfdc/dfdc_train_part_01/dfdc_train_part_1/qjdtgggqym.mp4'

metadata_path = '/home/deepak/DEEPFAKES_IITBHILAI/dfdc/dfdc_train_part_01/dfdc_train_part_1/metadata.json'

with open(metadata_path,'r') as f:
	metadata = json.load(f)

# print(metadata)

out_dir = './test_output/'
bbox_json = './test_output/test_boxes/qjdtgggqym.json'


frames_df = read_frames_df()
# frames_df = extract_ori_crops_and_landmarks(real_path, out_dir + 'real_frames/', bbox_json, frames_df, 5)
# frames_df = extract_fake_crops(fake_path, out_dir + 'fake_frames/', bbox_json, frames_df, 5)
# frames_df.to_csv('./test_output/frames_df.csv',index=False)

# extract_diff_masks(frames_df)
# print(frames_df.head())


# extract_landmarks(frames_df)
img = cv2.imread("./test_output/real_frames/qjdtgggqym_149.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
lmark = np.load("./test_output/landmarks/qjdtgggqym_149.png.npy",)
lmark = lmark[0][0]
# print(lmark[0][0])
# print(lmark)

f,ax=plt.subplots(1, 4, figsize=(15,15))

ax[0].imshow(blackout_convex_hull(img))
ax[1].imshow(blend_original(img))
ax[2].imshow(remove_landmark(img,lmark))
ax[3].imshow(img)

plt.show()

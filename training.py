from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
import os
import random
from augmentations import remove_landmark, blackout_convex_hull
from augmentations import prepare_bit_masks
from albumentations.augmentations.functional import rot90
from albumentations.pytorch.functional import img_to_tensor


class ClassifierDataset(Dataset):

	def __init__(self,
				data_path='./test_output/',
				fold=0,
				label_smoothing=0.01,
				padding_part=3,
				hardcore=True,
				crops_dir="frames",
				folds_csv="frames_df.csv",
				normalize={"mean": [0.485, 0.456, 0.406],
							"std": [0.229, 0.224, 0.225]},
				rotation=False,
				mode="train",
				reduce_val=True,
				oversample_real=True,
				transforms=None
				):

		super().__init__()
		self.data_root = data_path
		self.fold = fold
		self.folds_csv = folds_csv
		self.mode = mode
		self.rotation = rotation
		self.padding_part = padding_part
		self.hardcore = hardcore
		self.crops_dir = crops_dir
		self.label_smoothing = label_smoothing
		self.normalize = normalize
		self.transforms = transforms
		self.df = pd.read_csv(os.path.join(self.data_root,self.folds_csv))
		self.oversample_real = oversample_real
		self.reduce_val = reduce_val
		self.data = self.df.values


	def __getitem__(self, index: int):

		while True:
			frame, label, real_frame, fold = self.data[index]

			try:

				if self.mode == "train":
					label = np.clip(label, self.label_smoothing, 1-self.label_smoothing)

				if label > 0.5:
					img_path = os.path.join(self.data_root, 'fake_frames', frame)
				else:
					img_path = os.path.join(self.data_root, 'real_frames', frame)
				image = cv2.imread(img_path, cv2.IMREAD_COLOR)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				mask = np.zeros(image.shape[:2], dtype=np.uint8)
				diff_path = os.path.join(self.data_root, "diff_masks", frame)

				try:
					msk = cv2.imread(diff_path, cv2.IMREAD_GRAYSCALE)
					if msk is not None:
						mask = msk

				except:
					print("diff_mask not found!")
					pass

				if self.mode=="train" and self.hardcore and not self.rotation:
					landmark_path = os.path.join(self.data_root, "landmarks", real_frame + ".npy")
					# print(landmark_path, os.path.exists(landmark_path))
					# return (None,None)
					if os.path.exists(landmark_path) and random.random() < 0.7:
						landmarks = np.load(landmark_path, allow_pickle = True)
						if landmarks is not None and landmarks[0] is not None:
							image = remove_landmark(image, landmarks[0][0])
					elif random.random() < 0.2:
						image = blackout_convex_hull(image)
					elif random.random() < 0.1:
						binary_mask = mask > 0.4 * 255
						masks = prepare_bit_masks((binary_mask * 1).astype(np.uint8))
						tries = 6
						current_try = 1
						print("here")
						while current_try < tries:
							bitmap_msk = random.choice(masks)
							if label < 0.5 or np.count_nonzero(mask * bitmap_msk) > 20:
								mask *= bitmap_msk
								image *= np.expand_dims(bitmap_msk, axis=-1)
								break
							current_try += 1

				valid_label = np.count_nonzero(mask[mask > 20]) > 32 or label < 0.5
				valid_label = 1 if valid_label else 0
				rotation = 0

				if self.transforms:
					data = self.transforms(image=image, mask=mask)
					image = data["image"]
					mask = data["mask"]

				if self.mode == "train" and self.hardcore and self.rotation:
					dropout = 0.8 if label > 0.5 else 0.6
					if self.rotation:
						dropout *= 0.7
					elif random.random() < dropout:
						pass

				if self.mode == "train" and self.rotation:
					rotation = random.randint(0, 3)
					image = rot90(image, rotation)

				image = img_to_tensor(image, self.normalize)

				return {"image": image, 
					"labels": np.array([label]), 
					"img_name": frame,
					"valid": valid_label,
					"rotations": rotation}

				

			except e:
				print(str(e))
				exit(0)

import numpy as np
import cv2
import os
import torch
from albumentations.pytorch.functional import img_to_tensor

def predict_video(model, vid_dir, num_frames=32):
	vid_name = os.path.split(vid_dir)[-1].split('.')[0]
	capture = cv2.VideoCapture(vid_dir)
	frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

	if frame_count <= 0:
		return 0

	frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int)
	ori_frames = []
	idxs_read = 0
	for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):

		ret = capture.grab()
		if not ret:
			break

		if frame_idx == frame_idxs[idxs_read]:
			ret, frame = capture.retrieve()
			if not ret or frame is None:
				break

			ori_frames.append(frame)
			idxs_read += 1

	# Finding face bounding boxes

	detector = MTCNN()
	bboxes, *_ = detector.detect(ori_frames, landmarks=False)

	# Extracting crops

	landmarks_json = {}

	frame_labels = []

	crops = []

	for idx, data in enumerate(zip(frame_idxs, bboxes, ori_frames)):
		f_id, bbox, frame = data
		
		try:
			bbox = bbox[0]

			xmin, ymin, xmax, ymax = [int(b) for b in bbox]
			w = xmax - xmin
			h = ymax - ymin
			p_h = h // 3
			p_w = w // 3

			H, W = frame.shape[:2]

			crop = frame[
				   max(ymin - p_h, 0):min(ymax + p_h, H),
				   max(xmin - p_w, 0):min(xmax + p_w, W),
				   ]

			normalize = {"mean": [0.485, 0.456, 0.406],"std": [0.229, 0.224, 0.225]}
			image = img_to_tensor(crop, normalize)

			crops.append(image.unsqeeze(dim=0))
			


			# lmark_t = [[i[0],i[1]] for i in lmark[0]]
		except:
			pass
	# print(json.dumps(landmarks_json))
	crops = torch.cat(crops, dim=0)
	model.eval()
	out_labels = model(crops)

	fakes = (out_labels >= 0.5)

	fakes_count = torch.sum(fakes)

	return fakes_count/fakes.shape[0]
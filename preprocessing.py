from typing import List
import numpy as np
import os
import cv2
from facenet_pytorch.models.mtcnn import MTCNN
from skimage.metrics._structural_similarity import structural_similarity
from utils import centerCrop


class Preprocessor:

    def __init__(self, pairs, num_frames):
        self.pairs = pairs
        self.num_frames = num_frames
        self.root_dir = None
        self.frames = None
        self.crops = None
        self.landmarks = None

    def extractCropsAndLandmarks(self, root_dir, num_frames):

        # Sampling frames from videos

        capture = cv2.VideoCapture(root_dir)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            return

        frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int)

        frames = []
        idxs_read = 0

        for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):

            ret = capture.grab()
            if not ret:
                break

            if frame_idx == frame_idxs[idxs_read]:
                ret, frame = capture.retrieve()
                if not ret or frame is None:
                    break

                frames.append(frame)
                idxs_read += 1

        # Finding face bounding boxes

        detector = MTCNN()
        bboxes, *_ = detector.detect(frames, landmarks=False)

        # Extracting crops

        crops = []

        for bbox, frame in zip(bboxes, frames):
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

            crops.append(crop)

        # Generating landmarks
        # crops = frames
        landmarks_batch = []

        for crop in crops:
            _, _, landmarks = detector.detect(crop, landmarks=True)
            landmarks = np.around(landmarks[0]).astype(np.int16)
            landmarks_batch.append(landmarks)

        return crops, landmarks_batch

    def extractDiffMasks(self, ori_path, fake_path):

        ori_crops, _ = self.extractCropsAndLandmarks(ori_path, self.num_frames)
        fake_crops, _ = self.extractCropsAndLandmarks(fake_path, self.num_frames)

        masks: List[None] = []

        for ori, fake in zip(ori_crops, fake_crops):
            common_dim = (min(ori.shape[0], fake.shape[0]), min(ori.shape[1], fake.shape[1]))

            ori_center = centerCrop(ori, common_dim)
            fake_center = centerCrop(fake, common_dim)

            d, mask = structural_similarity(ori_center, fake_center, multichannel=True, full=True)
            mask = 1 - mask
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            masks.append(mask)

        return masks, ori_crops, fake_crops

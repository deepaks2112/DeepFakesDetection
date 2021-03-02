from typing import List
import json
import numpy as np
import os
import cv2
from facenet_pytorch.models.mtcnn import MTCNN
from skimage.metrics._structural_similarity import structural_similarity
from utils import centerCrop
import pandas as pd
from tqdm import tqdm



def read_frames_df():
    try:
        frames_df = pd.read_csv('./test_output/frames_df.csv')

    except:
        frames_df = pd.DataFrame(columns=['Frame', 'Label', 'Real Frame'])

    return frames_df


def extract_ori_crops_and_bboxes(vid_dir,out_dir,bbox_json,frames_df,num_frames=32):
    
    vid_name = os.path.split(vid_dir)[-1].split('.')[0]
    capture = cv2.VideoCapture(vid_dir)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        return

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

            crop_uid = vid_name + "_" + str(f_id) + ".png"
            cv2.imwrite(os.path.join(out_dir,crop_uid), crop)
            frames_df = frames_df.append({'Label': 0, 'Frame': crop_uid, 'Real Frame': crop_uid}, ignore_index=True)

            # lmark_t = [[i[0],i[1]] for i in lmark[0]]
            landmarks_json[crop_uid] = [int(b) for b in bbox]
        except:
            pass
    # print(json.dumps(landmarks_json))
    with open(os.path.join('.',bbox_json),'w') as fout:
        json.dump(landmarks_json, fout)

    return frames_df
    # frames_df.to_csv('./test_output/frames_df.csv',index=False)

def extract_fake_crops(vid_dir, out_dir, bbox_json, frames_df, num_frames = 32):

    with open(bbox_json,'r') as f:
        bbox_json_file = json.load(f)

    vid_name = os.path.split(vid_dir)[-1].split('.')[0]
    capture = cv2.VideoCapture(vid_dir)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        return frames_df

    frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int)

    fake_frames = []
    idxs_read = 0

    for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):

        ret = capture.grab()
        if not ret:
            break

        if frame_idx == frame_idxs[idxs_read]:
            ret, frame = capture.retrieve()
            if not ret or frame is None:
                break

            fake_frames.append(frame)
            idxs_read += 1

    ori_vid_name = os.path.split(bbox_json)[-1].split('.')[0]
    
    for frame_idx, frame in zip(frame_idxs, fake_frames):
        
        key = ori_vid_name + "_" + str(frame_idx) + ".png"
        bbox = bbox_json_file.get(key,[])

        if bbox==[]:
            continue

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

        crop_uid = vid_name + "_" + str(frame_idx) + ".png"
        cv2.imwrite(os.path.join(out_dir,crop_uid), crop)
        frames_df = frames_df.append({'Label': 1, 'Frame': crop_uid, 'Real Frame': ori_vid_name + "_" + str(frame_idx) + ".png"}, ignore_index=True)
    # frames_df.to_csv('./test_output/frames_df.csv',index=False)

    return frames_df

def extract_diff_masks(frames_df):

    real_frames = './test_output/real_frames/'
    fake_frames = './test_output/fake_frames/'
    diff_masks_dir = './test_output/diff_masks/'
    
    for i in tqdm(range(frames_df.shape[0])):

        if frames_df['Label'].values[i] == 1:

            real_frame = frames_df['Real Frame'].values[i]
            fake_frame = frames_df['Frame'].values[i]

            
            try:
                frame1 = cv2.imread(real_frames + real_frame)
                frame2 = cv2.imread(fake_frames + fake_frame)


                d, mask = structural_similarity(frame1, frame2, multichannel = True, full = True)
                mask = 1 - mask
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                mask_uid = fake_frame
                cv2.imwrite(os.path.join(diff_masks_dir, fake_frame), mask)

            except:
                pass


def extract_landmarks(frames_df):

    real_df = frames_df[frames_df['Label'] == 0]

    landmark_path = './test_output/landmarks/'
    real_path = './test_output/real_frames/'

    detector = MTCNN()

    ori_frames = []

    for i in tqdm(range(real_df.shape[0])):

        try:
            tmp = cv2.imread(real_path + real_df['Frame'].values[i])
            bboxes, _, landmarks = detector.detect([tmp], landmarks=True)
            np.save(landmark_path + real_df['Frame'].values[i], landmarks)
        except:
            pass


def generate_folds(frames_df, num_folds):

    n = frames_df.shape[0]

    folds = [i % num_folds for i in range(n)]
    np.random.shuffle(folds)
    
    frames_df['Fold'] = folds
    return frames_df

'''class Preprocessor:

    def __init__(self, pairs, num_frames):
        self.pairs = pairs
        self.num_frames = num_frames
        self.root_dir = None
        self.frames = None
        self.crops = None
        self.landmarks = None

    def extractCropsAndLandmarks(self, ori_path, fake_path, num_frames):

        # Sampling frames from videos

        capture = cv2.VideoCapture(ori_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            return

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

        ori_crops = []

        for bbox, frame in zip(bboxes, ori_frames):
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

            ori_crops.append(crop)

        # Cropping fake frames
        capture = cv2.VideoCapture(fake_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            return

        frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=np.int)

        fake_frames = []
        idxs_read = 0

        for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):

            ret = capture.grab()
            if not ret:
                break

            if frame_idx == frame_idxs[idxs_read]:
                ret, frame = capture.retrieve()
                if not ret or frame is None:
                    break

                fake_frames.append(frame)
                idxs_read += 1

        fake_crops = []

        for bbox, frame in zip(bboxes, fake_frames):
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

            fake_crops.append(crop)
        # Generating landmarks
        # crops = frames
        landmarks_batch = []

        for crop in ori_crops:
            _, _, landmarks = detector.detect(crop, landmarks=True)
            landmarks = np.around(landmarks[0]).astype(np.int16)
            landmarks_batch.append(landmarks)

        return ori_crops, fake_crops, landmarks_batch

    def extractDiffMasks(self, ori_path, fake_path):

        ori_crops, fake_crops, _ = self.extractCropsAndLandmarks(ori_path, fake_path, self.num_frames)

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

        return masks, ori_crops, fake_crops'''

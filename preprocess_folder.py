import sys
from preprocessing import read_frames_df, extract_ori_crops_and_bboxes
from preprocessing import extract_fake_crops, extract_diff_masks, extract_landmarks
from preprocessing import generate_folds
from utils import get_video_pairs, get_ori_videos
import json
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--metadata', type=str, required=True)
parser.add_argument('--frames', type=int, required=True)
parser.add_argument('--folds', type=int, required=True)

args = parser.parse_args()

videos_path = args.dir
metadata_path = args.metadata
num_frames = args.frames
num_folds = args.folds

print(videos_path, metadata_path)
print("Continue?")
a=int(input())

out_dir = './test_output/'

with open(metadata_path, 'r') as f:
	metadata = json.load(f)

frames_df = read_frames_df()

ori_videos = get_ori_videos(metadata)
video_pairs = get_video_pairs(metadata)

ori_set = set()
for p in video_pairs:
	ori_set.add(p[1])

ori_videos += list(ori_set)

print("Extracting real frames...")

for real in tqdm(ori_videos):

	real_path = os.path.join(videos_path, real)
	bbox_json = os.path.join('./test_output/test_boxes', real[:-4] + '.json')
	frames_df = extract_ori_crops_and_bboxes(real_path, out_dir + 'real_frames/', bbox_json, frames_df, num_frames)


print("Extracting fake frames...")

for fake, real in tqdm(video_pairs):

	fake_path = os.path.join(videos_path, fake)
	real_path = os.path.join(videos_path, real)
	bbox_json = os.path.join('./test_output/test_boxes', real[:-4] + '.json')
	frames_df = extract_fake_crops(fake_path, out_dir + 'fake_frames/', bbox_json, frames_df, num_frames)

print("Generating folds...")
frames_df = generate_folds(frames_df, num_folds)
frames_df.to_csv(out_dir + 'frames_df.csv', index=False)

print("Extracting diff masks...")
extract_diff_masks(frames_df)

print("Extracting landmarks...")
extract_landmarks(frames_df)


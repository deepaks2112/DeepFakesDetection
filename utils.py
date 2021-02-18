import numpy as np

def centerCrop(img, size):

    curr_size = img.shape[:2]

    diff_h = (curr_size[0] - size[0]) // 2
    diff_w = (curr_size[1] - size[1]) // 2

    return img[diff_h:diff_h + size[0], diff_w:diff_w + size[1]]


def get_video_pairs(metadata):
	
	video_pairs = []

	for k,v in metadata.items():
		if v['label'] == 'FAKE':
			video_pairs.append((k,v['original']))

	return video_pairs


def get_ori_videos(metadata):

	ori_videos = []

	for k,v in metadata.items():
		if v['label'] == 'REAL':
			ori_videos.append(k)

	return ori_videos
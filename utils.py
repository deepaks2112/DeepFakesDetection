import numpy as np

def centerCrop(img, size):

    curr_size = img.shape[:2]

    diff_h = (curr_size[0] - size[0]) // 2
    diff_w = (curr_size[1] - size[1]) // 2

    return img[diff_h:diff_h + size[0], diff_w:diff_w + size[1]]
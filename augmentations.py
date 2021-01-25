import dlib
import skimage.draw
from skimage import measure
import numpy as np
import random
from scipy.ndimage import binary_dilation, binary_erosion
import cv2
import math


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def prepare_bit_masks(mask):
    h, w = mask.shape
    mid_w = w // 2
    mid_h = h // 2
    masks = []

    ones = np.ones_like(mask)
    ones[:mid_h] = 0
    masks.append(ones)

    ones = np.ones_like(mask)
    ones[mid_h:] = 0
    masks.append(ones)

    ones = np.ones_like(mask)
    ones[:, :mid_w] = 0
    masks.append(ones)

    ones = np.ones_like(mask)
    ones[:, mid_w:] = 0
    masks.append(ones)

    ones = np.ones_like(mask)
    ones[:mid_h, :mid_w] = 0
    ones[mid_h:, mid_w:] = 0
    masks.append(ones)

    ones = np.ones_like(mask)
    ones[:mid_h, mid_w:] = 0
    ones[mid_h:, :mid_w] = 0
    masks.append(ones)

    return masks


def blackout_convex_hull(img):
    img = img.copy()
    try:
        rect = detector(img)[0]
        sp = predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        outline=landmarks[[*range(17), *range(26, 16, -1)]]
        Y, X = skimage.draw.polygon(outline[:,1], outline[:,0])
        cropped_img = np.zeros(img.shape[:2], dtype=np.uint8)
        cropped_img[Y, X] = 1

        y, x = measure.centroid(cropped_img)
        y = int(y)
        x = int(x)
        first = random.random() > 0.5
        if random.random() > 0.5:
            if first:
                cropped_img[:y, :] = 0
            else:
                cropped_img[y:, :] = 0
        else:
            if first:
                cropped_img[:, :x] = 0
            else:
                cropped_img[:, x:] = 0

        img[cropped_img > 0] = 0

    except:
        pass
    return img


def remove_eyes(img, landmarks):
    img = img.copy()

    (x1, y1), (x2, y2) = landmarks[:2]

    mask = np.zeros_like(img[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    img[line, :] = 0
    return img

def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def remove_nose(img, landmarks):
    img = img.copy()

    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    x4, y4 = int((x1 + x2)/2), int((y1 + y2)/2)
    
    mask = np.zeros_like(img[..., 0])

    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    img[line, :] = 0
    return img

def remove_mouth(img, landmarks):
    img = img.copy()

    (x1, y1), (x2, y2) = landmarks[-2:]
    
    mask = np.zeros_like(img[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)

    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)

    line = binary_dilation(line, iterations=dilation)

    img[line, :] = 0
    return img


def remove_landmark(img, landmarks):
    if random.random() > 0.5:
        img = remove_eyes(img, landmarks)
    elif random.random() > 0.5:
        img = remove_mouth(img, landmarks)
    elif random.random() > 0.5:
        img = remove_nose(img, landmarks)
    return img


def blend_original(img):
    img = img.copy()

    h, w = img.shape[:2]
    rect = detector(img)

    if len(rect) == 0:
        return img
    else:
        rect = rect[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    outline = landmarks[[*range(17), *range(26,16,-1)]]
    Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
    raw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    raw_mask[Y, X] = 1
    face = img * np.expand_dims(raw_mask, -1)

    h1 = random.randint(h - h // 2, h + h // 2)
    w1 = random.randint(w - w // 2, w + w // 2)
    while abs(h1 - h) < h // 3 and abs(w1 - w) < w // 3:
        h1 = random.randint(h - h // 2, h + h // 2)
        w1 = random.randint(w - w // 2, w + w // 2)

    face = cv2.resize(face, (w1, h1), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]))
    face = cv2.resize(face, (w, h), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]))

    raw_mask = binary_erosion(raw_mask, iterations=random.randint(4, 10))
    img[raw_mask, :] = face[raw_mask, :]

    # if random.random() < 0.2:
    #     img = OneOf([GaussianBlur])
    return img

# def blackout_random(img, mask, label):



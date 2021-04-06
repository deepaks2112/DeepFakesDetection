import dlib
import skimage.draw
from skimage import measure
import numpy as np
import random
from scipy.ndimage import binary_dilation, binary_erosion
import cv2
import math
from albumentations import DualTransform
from albumentations.augmentations.functional import crop


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size

    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):

    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, always_apply=False, p=1):

        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up


    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down, interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)


    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")
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

        # if random.random >= 0.5:
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



def thick_outline_polygon(mask, Y, X, thickness):
  mask=mask.copy()
  for y,x in zip(Y,X):
    mask[y][x]=np.ones(3)
    for i in range(max(0,y-thickness//2),y):
      mask[i][x]=np.ones(3)
    for i in range(y,min(mask.shape[0],y+thickness//2)):
      mask[i][x]=np.ones(3)
    for i in range(max(0,x-thickness//2),x):
      mask[y][i]=np.ones(3)
    for i in range(x,min(mask.shape[1],x+thickness//2)):
      mask[y][i]=np.ones(3)
  
  # for i in range(mask.shape[0]):
  #   for j in range(mask.shape[1]):
  #     if mask[i][j]>0:
  #       mask[i][j]=1
  
  return mask


def get_convex_hull_outline(img, thickness=60):
    img=img.copy()
    try:
        rect = detector(img)[0]
        sp = predictor(img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        outline=landmarks[[*range(17), *range(26, 16, -1)]]

        # if random.random >= 0.5:
        Y, X = skimage.draw.polygon_perimeter(outline[:,1], outline[:,0])
        mask=np.zeros(img.shape, dtype=np.float32)
    # print(mask.shape)
        mask2=thick_outline_polygon(mask, Y, X, 60)

        img2=(img*mask2)

    except:
        img2=img
        pass
    return img2
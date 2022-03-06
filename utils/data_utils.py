import math
import os
import random

import yaml
import numpy as np
import pandas as pd
import cv2
import copy
import matplotlib.pyplot as plt
import torch
from xml.etree import ElementTree
from PIL import Image
from sklearn.model_selection import train_test_split

labels = ['with_mask', 'mask_weared_incorrect', 'without_mask']


# load images in a mosaic
def load_mosaic(self, index):
    labels4 = []
    s = 640
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # center x, y
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]
    for i, index in enumerate(indices):
        img, _, (h, w) = load_image(self, index)

        if i == 0:
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

    img4, labels4 = random_affine(img4, labels4, degrees=self.hyp['degrees'], translate=self.hyp['translate'],
                                  scale=self.hyp['scale'], shear=self.hyp['shear'], border=self.mosaic_border)
    return img4, labels4


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=(0, 0)):
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    # rotation and scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[1] + border[1]
    T[1, 2] = random.uniform(-translate, translate) * img.shape[0] + border[0]

    # shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # combined rotation matrix
    M = S @ T @ R
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 20)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def cache_labels(self, cache_path):
    x = {}  # dict
    for (img, label) in zip(self.img_files, self.ann_files):
        try:
            l = []
            image = Image.open(img)
            image.verify()
            shape = (image.size[0], image.size[1])
            if os.path.isfile(label):
                l = np.array([x for x in extract_annotation_file(label, shape)], dtype=np.float32)
            if len(l) == 0:
                l = np.zeros((0, 5), dtype=np.float32)
            x[img] = [l, shape]
        except Exception as e:
            x[img] = None
            print('WARNING: %s: %s' % (img, e))

    x['hash'] = get_hash(self.ann_files + self.img_files)
    torch.save(x, cache_path)
    return x


def get_hash(files):
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), scale_fill=False, scaleup=True):
    # 이미지를 사각형으로 resize(32 pixel)
    shape = img.shape[:2]

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def load_image(self, index):
    img = self.imgs[index]
    if img is None:
        img_name = self.img_files[index]
        img = cv2.imread(img_name)
        h0, w0 = img.shape[:2]
        r = 640 / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 and self.transform else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]
    else:
        '''print(self.imgs[index])
        print(self.img_hw0[index])
        print(self.img_hw[index])'''
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]


def get_yolo_format(pic_width, pic_height, x_min, y_min, x_max, y_max):
    x_center = (x_max + x_min) / (2 * pic_width)
    y_center = (y_max + y_min) / (2 * pic_height)
    width = (x_max - x_min) / pic_width
    height = (y_max - y_min) / pic_height
    return x_center, y_center, width, height


def extract_annotation_file(ann_file, shape):
    tree = ElementTree.parse(ann_file)
    boxes = list()

    pic_width, pic_height = shape[0], shape[1]

    for box in tree.findall('.//object'):
        cls = labels.index(box.find('name').text)
        xmin = int(box.find('bndbox/xmin').text)
        ymin = int(box.find('bndbox/ymin').text)
        xmax = int(box.find('bndbox/xmax').text)
        ymax = int(box.find('bndbox/ymax').text)

        x_center, y_center, box_width, box_height = get_yolo_format(pic_width, pic_height, xmin, ymin, xmax, ymax)

        boxes.append([cls, x_center, y_center, box_width, box_height])

    return boxes


def split_data(img_files, show_cnt=5):
    image_train, image_else = train_test_split(img_files, test_size=0.2)
    image_val, image_test = train_test_split(image_else, test_size=show_cnt / len(image_else))

    return [0, len(image_train), len(image_val), len(image_test)]


def create_yaml(data_dir):
    yaml_file = data_dir + '/config.yaml'

    yaml_data = dict(
        nc=len(labels),
        names=labels
    )

    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_data, f, explicit_start=True, default_flow_style=False)


import os
import re
from pathlib import Path
from tqdm import tqdm
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import xywh2xyxy, xyxy2xywh
from utils.data_utils import get_yolo_format, extract_annotation_file, split_data, create_yaml, get_hash, \
    cache_labels, random_affine, load_image, load_mosaic, augment_hsv

labels = ['with_mask', 'mask_weared_incorrect', 'without_mask']


class Dataset(Dataset):
    def __init__(self, data_dir, idx1, idx2, hyp, augment=None):
        super(Dataset, self).__init__()
        self.hyp = hyp
        img_files = [data_dir + '/images/' + img_file for img_file in os.listdir(os.path.join(data_dir, 'images'))
                     if img_file[-4:] == '.png']
        ann_files = [img_file.replace('images', 'annotations')[:-4] + '.xml' for img_file in img_files]
        split = split_data(img_files)
        img_files.sort(key=lambda x: int(re.sub('[^0-9]', '', x)))
        ann_files.sort(key=lambda x: int(re.sub('[^0-9]', '', x)))

        self.img_files = img_files[split[idx1]:split[idx2]+split[idx1]]
        self.ann_files = ann_files[split[idx1]:split[idx2]+split[idx1]]
        self.file_dir = data_dir
        self.augment = augment
        self.mosaic_border = (-640 // 2, -640 // 2)
        cache_path = str(Path(ann_files[0]).parent) + 'cache'
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)
            if cache['hash'] != get_hash(self.ann_files + self.img_files):
                cache = cache_labels(self, cache_path)
        else:
            cache = cache_labels(self, cache_path)

        # Get labels
        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        gb = 0
        pbar = tqdm(range(len(self.img_files)), desc='caching image')
        self.imgs = [None] * self.__len__()
        self.img_hw0 = [None] * self.__len__()
        self.img_hw = [None] * self.__len__()
        for i in pbar:
            self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)
            gb += self.imgs[i].nbytes
            pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        image, label = load_mosaic(self, index)
        shapes = None
        if self.augment:
            augment_hsv(image, hgain=self.hyp['hsv_h'], sgain=self.hyp['hsv_s'], vgain=self.hyp['hsv_v'])

        nL = len(label)
        if nL:
            label[:, 1:5] = xyxy2xywh(label[:, 1:5])

            label[:, [2, 4]] /= image.shape[0]
            label[:, [1, 3]] /= image.shape[1]

        # random left-right flip
        if self.augment:
            if random.random() < 0.5:
                image = np.fliplr(image)
                if nL:
                    label[:, 1] = 1 - label[:, 1]

        label_out = torch.zeros((nL, 6))
        if nL:
            label_out[:, 1:] = torch.from_numpy(label)

        # Convert
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        data = {'image': torch.from_numpy(image), 'label': label_out}

        return data



# 이미지 알부멘테이션 후 라벨 데이터 다시 전처리 pix2백분율
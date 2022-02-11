import os
import re
import torch
import torch.nn as nn
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from utils.data_utils import get_yolo_format, extract_annotation_file, split_data, create_yaml

labels = ['with_mask', 'mask_weared_incorrect', 'without_mask']

class TrainSet(Dataset):
    def __init__(self, data_dir, transform=None):
        super(TrainSet, self).__init__()
        img_files = [img_file for img_file in os.listdir(os.path.join(data_dir, 'images'))
                     if img_file[-4:] == '.png']
        ann_files = [img_file[:-4] + '.xml' for img_file in img_files]
        img_files.sort(key=lambda x: int(re.sub('[^0-9]', '', x)))
        ann_files.sort(key=lambda x: int(re.sub('[^0-9]', '', x)))
        split = split_data(img_files)
        img_files = img_files[:split[0]+1]
        ann_files = ann_files[:split[0]+1]
        print(img_files[-5:], ann_files[-5:])
        images = pd.Series(img_files, name='images')
        annots = pd.Series(ann_files, name='annots')
        df = pd.concat([images, annots], axis=1)
        self.data_frame = pd.DataFrame(df)
        self.file_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame[..., 0])

    def __getitem__(self, index):
        label = torch.tensor(extract_annotation_file(self.data_frame[index, 1], self.file_dir))

        image = Image.open(os.path.join(self.file_dir, 'images') + '/' + self.data_frame[index, 0]).convert('RGB')

class ValSet(Dataset):
    def __init__(self, data_dir, transform=None):
        super(ValSet, self).__init__()
        img_files = [img_file for img_file in os.listdir(os.path.join(data_dir, 'images'))
                    if img_file[-4:] == '.png']
        ann_files = [img_file[:-4] + '.xml' for img_file in img_files]
        img_files.sort(key=lambda x: int(re.sub('[^0-9]', '', x)))
        ann_files.sort(key=lambda x: int(re.sub('[^0-9]', '', x)))
        split = split_data(img_files)
        img_files = img_files[split[0]:split[0] + split[1] + 1]
        ann_files = ann_files[split[0]:split[0] + split[1] + 1]
        print(img_files[-5:], ann_files[-5:])
        images = pd.Series(img_files, name='images')
        annots = pd.Series(ann_files, name='annots')
        df = pd.concat([images, annots], axis=1)
        self.data_frame = pd.DataFrame(df)
        self.file_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame[..., 0])

    def __getitem__(self, index):
        label = torch.tensor(extract_annotation_file(self.data_frame[index, 1], self.file_dir))

        image = Image.open(os.path.join(self.file_dir, 'images') + '/' + self.data_frame[index, 0]).convert('RGB')


class TestSet(Dataset):
    def __init__(self, data_dir, transform=None):
        super(TestSet, self).__init__()
        img_files = [img_file for img_file in os.listdir(os.path.join(data_dir, 'images'))
                     if img_file[-4:] == '.png']
        ann_files = [img_file[:-4] + '.xml' for img_file in img_files]
        img_files.sort(key=lambda x: int(re.sub('[^0-9]', '', x)))
        ann_files.sort(key=lambda x: int(re.sub('[^0-9]', '', x)))
        split = split_data(img_files)
        img_files = img_files[split[0]:split[0] + split[1] + 1]
        ann_files = ann_files[split[0]:split[0] + split[1] + 1]
        print(img_files[-5:], ann_files[-5:])
        images = pd.Series(img_files, name='images')
        annots = pd.Series(ann_files, name='annots')
        df = pd.concat([images, annots], axis=1)
        self.data_frame = pd.DataFrame(df)
        self.file_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame[..., 0])

    def __getitem__(self, index):
        label = torch.tensor(extract_annotation_file(self.data_frame[index, 1], self.file_dir))

        image = Image.open(os.path.join(self.file_dir, 'images') + '/' + self.data_frame[index, 0]).convert('RGB')





a = TrainSet('/Users/sinmugyeol/dataset/facemask')



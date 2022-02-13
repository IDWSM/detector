import os
import yaml
import pandas as pd
from xml.etree import ElementTree
from PIL import Image
from sklearn.model_selection import train_test_split

labels = ['with_mask', 'mask_weared_incorrect', 'without_mask']

def get_yolo_format(pic_width, pic_height, x_min, y_min, x_max, y_max):
    x_center = (x_max + x_min) / (2 * pic_width)
    y_center = (y_max + y_min) / (2 * pic_height)
    width = (x_max - x_min) / pic_width
    height = (y_max - y_min) / pic_height
    return x_center, y_center, width, height


def extract_annotation_file(file_name, path):
    tree = ElementTree.parse(os.path.join(path, 'annotations') + '/' + file_name)
    boxes = list()

    img = Image.open(os.path.join(path, 'images') + '/' + file_name[:-4] + '.png')
    pic_width, pic_height = img.size[0], img.size[1]

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

    return [len(image_train), len(image_val), len(image_test)]

def create_yaml(data_dir):
    yaml_file = data_dir + '/config.yaml'

    yaml_data = dict(
        nc=len(labels),
        names=labels
    )

    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_data, f, explicit_start=True, default_flow_style=False)



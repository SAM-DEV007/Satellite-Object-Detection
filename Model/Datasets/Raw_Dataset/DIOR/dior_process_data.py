from pathlib import Path

import numpy as np

import os
import cv2


def tile(bounds, x_start, y_start, size):
    _class, x_min, y_min, x_max, y_max = bounds
    x_min, y_min, x_max, y_max = x_min - x_start, y_min - y_start, x_max - x_start, y_max - y_start

    if (x_min > size) or (x_max < 0.0) or (y_min > size) or (y_max < 0.0):
        return None
    
    x_max_trunc = min(x_max, size) 
    x_min_trunc = max(x_min, 0) 
    if (x_max_trunc - x_min_trunc) / (x_max - x_min) < 0.3:
        return None

    y_max_trunc = min(y_max, size) 
    y_min_trunc = max(y_min, 0) 
    if (y_max_trunc - y_min_trunc) / (y_max - y_min) < 0.3:
        return None
        
    x_center = (x_min_trunc + x_max_trunc) / 2.0 / size
    y_center = (y_min_trunc + y_max_trunc) / 2.0 / size
    x_extend = (x_max_trunc - x_min_trunc) / size
    y_extend = (y_max_trunc - y_min_trunc) / size
    
    return (_class, x_center, y_center, x_extend, y_extend)


# https://stackoverflow.com/a/64097592/16660603
def yolobbox2bbox(coords: list, size: int) -> list:
    x, y, w, h = coords

    x1 = int((x - w / 2) * size)
    x2 = int((x + w / 2) * size)
    y1 = int((y - h / 2) * size)
    y2 = int((y + h / 2) * size)
    
    if x1 < 0:
        x1 = 0
    if x2 > size - 1:
        x2 = size - 1
    if y2 < 0:
        y1 = 0
    if y2 > size - 1:
        y2 = size - 1

    return [x1, y1, x2, y2]


def load_img_annotation(load_data_path: str, img_path: str, labels_path: str) -> tuple:
    img_paths, labels_paths = [], []
    img, lbl = os.listdir(img_path), os.listdir(labels_path)
    img.sort(); lbl.sort()

    with open(load_data_path, 'r') as f:
        for name in f:
            name = name.strip()
            name = name.replace('\n', '') + '.jpg'
            if name in img: img_paths.append(rf'{img_path}\{name}')
            else: continue

            name = name.replace(".jpg", ".txt")
            if name in lbl: labels_paths.append(rf'{labels_path}\{name}')
            else: img_paths.remove(rf'{img_path}\{name}') # Fail safe
    
    return (img_paths, labels_paths)


if __name__ == '__main__':
    tile_size = 512
    tile_overlap = 64

    data_path = str(Path.cwd()) + r'\Model\Datasets\Raw_Dataset\DIOR\Raw'
    save_path = str(Path.cwd()) + r'\Model\Datasets\Raw_Dataset\DIOR\Processed'

    train_img_name_path = data_path + r'\imageSets\train.txt'
    val_img_name_path = data_path + r'\imageSets\val.txt'
    test_img_name_path = data_path + r'\imageSets\test.txt'

    img_path = data_path + r'\images'
    labels_path = data_path + r'\labels'
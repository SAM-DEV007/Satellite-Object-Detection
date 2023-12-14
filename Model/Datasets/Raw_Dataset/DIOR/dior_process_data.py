from pathlib import Path
from tqdm import tqdm

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

    if x_max == x_min or y_max == y_min: return None

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
    _c, x, y, w, h = coords

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

    return [int(_c), x1, y1, x2, y2]


def load_annotation(path: str) -> list:
    with open(path, 'r') as f:
        ann = [list(map(float, line.replace('\n', '').split())) for line in f]

    return ann


def load_img_path(load_data_path: str, img_path: str, labels_path: str) -> tuple:
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


def process_data(mode: str, load_data_list: tuple, save_path: str) -> None:
    img, label = load_img_path(*load_data_list)
    assert len(img) == len(label), 'Image and label length mismatch'

    default_size = 800
    tile_size = 512
    tile_overlap = 64

    output_paths = [save_path + rf'\{mode}\images', save_path + rf'\{mode}\labels']
    for _path in output_paths:
        if not os.path.isdir(_path):
            os.makedirs(_path)

    for i in tqdm(range(len(img))):
        image_path = img[i]
        label_path = label[i]

        _img_name = os.path.basename(image_path)
        _annot_name = os.path.basename(label_path)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (default_size, default_size))
        size = image.shape[0]

        annotation_list = load_annotation(label_path)
        coords_list = [yolobbox2bbox(al, size=size) for al in annotation_list]

        x_tiles = (size + tile_size - tile_overlap - 1) // (tile_size - tile_overlap)
        y_tiles = (size + tile_size - tile_overlap - 1) // (tile_size - tile_overlap)

        for x in range(x_tiles):
            for y in range(y_tiles):
                x_end = min((x + 1) * tile_size - tile_overlap * (x != 0), size)
                x_start = x_end - tile_size
                y_end = min((y + 1) * tile_size - tile_overlap * (y != 0), size)
                y_start = y_end - tile_size

                save_tile_path = output_paths[0] + rf'\{_img_name.split(".")[0]}_{x_start}_{y_start}.jpg'
                save_label_path = output_paths[1] + rf'\{_annot_name.split(".")[0]}_{x_start}_{y_start}.txt'

                cut_tile = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)
                cut_tile[0:tile_size, 0:size, :] = image[y_start:y_end, x_start:x_end, :]
                cv2.imwrite(save_tile_path, cut_tile)

                found_tags = [
                    tile(bounds, x_start, y_start, tile_size)
                    for bounds in coords_list]
                found_tags = [el for el in found_tags if el is not None]

                with open(save_label_path, 'w+') as f:
                    for tags in found_tags:
                        f.write(' '.join(str(x) for x in tags) + '\n')


if __name__ == '__main__':
    data_path = str(Path.cwd()) + r'\Model\Datasets\Raw_Dataset\DIOR\Raw'
    save_path = str(Path.cwd()) + r'\Model\Datasets\Raw_Dataset\DIOR\Processed'

    train_img_name_path = data_path + r'\imageSets\train.txt'
    val_img_name_path = data_path + r'\imageSets\val.txt'
    test_img_name_path = data_path + r'\imageSets\test.txt'

    img_path = data_path + r'\images'
    labels_path = data_path + r'\labels'

    #Train
    process_data('train', (train_img_name_path, img_path, labels_path), save_path)
    # Val
    process_data('val', (val_img_name_path, img_path, labels_path), save_path)
    # Test
    process_data('test', (test_img_name_path, img_path, labels_path), save_path)
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
    c, x, y, w, h = coords

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

    return [int(c), x1, y1, x2, y2]


def load_annotation(path: str) -> list:
    with open(path, 'r') as f:
        # [Class, cx, xy, w, h] 
        ann = [list(map(float, line.replace('\n', '').split())) for line in f]

    return ann


if __name__ == '__main__':
    name = 'himanshu'

    tile_size = 576
    tile_overlap = 64

    data_path = str(Path.cwd()) + r'\TILES\InputData'
    image_path = data_path + rf'\{name}.jpeg'
    annotation_path = data_path + rf'\{name}\obj_train_data\{name}.txt'

    output_folder_path = str(Path.cwd()) + r'\TILES\OutputData'
    output_paths = [output_folder_path + r'\images', output_folder_path + r'\labels']

    for _path in output_paths:
        if not os.path.isdir(_path):
            os.makedirs(_path)
    
    image = cv2.imread(image_path)
    size = image.shape[0]
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    annotation_list = load_annotation(annotation_path)
    coords_list = [yolobbox2bbox(al, size=size) for al in annotation_list]
    
    x_tiles = (size + tile_size - tile_overlap - 1) // (tile_size - tile_overlap)
    y_tiles = (size + tile_size - tile_overlap - 1) // (tile_size - tile_overlap)

    # print(x_tiles, y_tiles)

    for x in range(x_tiles):
        for y in range(y_tiles):
            x_end = min((x + 1) * tile_size - tile_overlap * (x != 0), size)
            x_start = x_end - tile_size
            y_end = min((y + 1) * tile_size - tile_overlap * (y != 0), size)
            y_start = y_end - tile_size

            # print(x_start, y_start, x_end, y_end)

            save_tile_path = output_paths[0] + rf'\{name}_{x_start}_{y_start}.jpg'
            save_label_path = output_paths[1] + rf'\{name}_{x_start}_{y_start}.txt'

            cut_tile = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)
            cut_tile[0:tile_size, 0:size, :] = image[y_start:y_end, x_start:x_end, :]
            cv2.imwrite(save_tile_path, cut_tile)

            found_tags = [
                tile(bounds, x_start, y_start, tile_size)
                for bounds in coords_list]
            found_tags = [el for el in found_tags if el is not None]

            # print(found_tags)

            with open(save_label_path, 'w+') as f:
                for tags in found_tags:
                    f.write(' '.join(str(x) for x in tags) + '\n')
from pathlib import Path
from tqdm.autonotebook import tqdm

import pandas as pd
import numpy as np

import cv2
import ast
import os


def cvt_to_py(x): 
    return ast.literal_eval(x.rstrip('\r\n'))


def getBounds(geometry):
    try: 
        arr = np.array(geometry).T
        xmin = np.min(arr[0])
        ymin = np.min(arr[1])
        xmax = np.max(arr[0])
        ymax = np.max(arr[1])
        return (xmin, ymin, xmax, ymax)
    except:
        return np.nan


def getWidth(bounds):
    try:
        (xmin, ymin, xmax, ymax) = bounds
        return np.abs(xmax - xmin)
    except:
        return np.nan


def getHeight(bounds):
    try: 
        (xmin, ymin, xmax, ymax) = bounds
        return np.abs(ymax - ymin)
    except:
        return np.nan


def tile(bounds, x_start, y_start, size):
    x_min, y_min, x_max, y_max = bounds
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
    
    return (8, x_center, y_center, x_extend, y_extend)


if __name__ == '__main__':
    data_path = str(Path.cwd()) + r'\Model\Datasets\Raw_Dataset\Airplane\Raw'
    save_path = str(Path.cwd()) + r'\Model\Datasets\Raw_Dataset\Airplane\Processed'
    
    img_list = Path(data_path).glob('images/*.jpg')

    df = pd.read_csv(data_path + r'\annotations.csv', converters={'geometry': cvt_to_py})
    df.loc[:,'bounds'] = df.loc[:,'geometry'].apply(getBounds)
    df.loc[:,'width'] = df.loc[:,'bounds'].apply(getWidth)
    df.loc[:,'height'] = df.loc[:,'bounds'].apply(getHeight)

    index = df['image_id'].unique()
    val_indexes = index[len(index)//5:len(index)*2//5]

    tile_size = 512
    tile_overlap = 64

    for mode in ('train', 'val'):
        output_paths = [save_path + rf'\{mode}\images', save_path + rf'\{mode}\labels']
        for _path in output_paths:
            if not os.path.isdir(_path):
                os.makedirs(_path)

    tiles_dir = {'train': Path(save_path + r'\train\images'), 'val': Path(save_path + r'\val\images')}
    labels_dir = {'train': Path(save_path + r'\train\labels'), 'val': Path(save_path + r'\val\labels')}
    
    for img_path in tqdm(img_list):
        image = cv2.imread(str(img_path))
        size = image.shape[0]

        img_labels = df[df["image_id"] == img_path.name]
        
        x_tiles = (size + tile_size - tile_overlap - 1) // (tile_size - tile_overlap)
        y_tiles = (size + tile_size - tile_overlap - 1) // (tile_size - tile_overlap)

        for x in range(x_tiles):
            for y in range(y_tiles):
                x_end = min((x + 1) * tile_size - tile_overlap * (x != 0), size)
                x_start = x_end - tile_size
                y_end = min((y + 1) * tile_size - tile_overlap * (y != 0), size)
                y_start = y_end - tile_size

                folder = 'val' if img_path.name in val_indexes else 'train'
                save_tile_path = tiles_dir[folder].joinpath(img_path.stem + "_" + str(x_start) + "_" + str(y_start) + ".jpg")
                save_label_path = labels_dir[folder].joinpath(img_path.stem + "_" + str(x_start) + "_" + str(y_start) + ".txt")

                cut_tile = np.zeros(shape=(tile_size, tile_size, 3), dtype=np.uint8)
                cut_tile[0:tile_size, 0:size, :] = image[y_start:y_end, x_start:x_end, :]
                cv2.imwrite(str(save_tile_path), cut_tile)

                found_tags = [
                    tile(bounds, x_start, y_start, tile_size)
                    for bounds in img_labels['bounds']]
                found_tags = [el for el in found_tags if el is not None]

                with open(save_label_path, 'w+') as f:
                    for tags in found_tags:
                        f.write(' '.join(str(x) for x in tags) + '\n')
    
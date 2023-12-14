from pathlib import Path
from tqdm import tqdm

import shutil
import os

import dior_process_data # Custom module


def process_data(mode: str, load_data_list: tuple, save_path: str) -> None:
    img, label = dior_process_data.load_img_path(*load_data_list)

    output_paths = [save_path + rf'\{mode}\images', save_path + rf'\{mode}\labels']
    for _path in output_paths:
        if not os.path.isdir(_path):
            os.makedirs(_path)
    
    for i in tqdm(range(len(img))):
        image_path = img[i]
        label_path = label[i]

        shutil.copy(image_path, output_paths[0])
        shutil.copy(label_path, output_paths[1])


if __name__ == '__main__':
    data_path = str(Path.cwd()) + r'\Model\Datasets\Raw_Dataset\DIOR\Raw'
    save_path = str(Path.cwd()) + r'\Model\Model_Data\Model_Dataset'

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
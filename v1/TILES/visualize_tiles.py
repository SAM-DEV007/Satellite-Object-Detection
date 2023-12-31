from pathlib import Path

import cv2
import os


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


def load_annotation(path: str) -> list:
    with open(path, 'r') as f:
        # [Class, cx, xy, w, h] 
        ann = [list(map(float, line.replace('\n', '').split())) for line in f]

    return ann


if __name__ == '__main__':
    name = 'himanshu'
    img_size = 576

    data_path = str(Path.cwd()) + r'\v1\TILES\OutputData'
    data_paths = [data_path + r'\images', data_path + r'\labels']

    annot_save_path = str(Path.cwd()) + r'\v1\TILES\TilesAnnotations'
    if not os.path.isdir(annot_save_path):
        os.makedirs(annot_save_path)

    img_data = os.listdir(data_paths[0])
    label_data = os.listdir(data_paths[1])

    for i in range(len(img_data)):
        image_path = data_paths[0] + rf'\{img_data[i]}'
        annotation_path = data_paths[1] + rf'\{label_data[i]}'

        image = cv2.imread(image_path)
        img_size = image.shape[0]
        annotation_list = load_annotation(annotation_path)

        for al in annotation_list:
            color = [(255, 0, 0), (0, 0, 255), (0, 255, 0)][int(al[0])]

            x, y, w, h = yolobbox2bbox(al[1:], size=img_size)
            cv2.rectangle(image, (x, y), (w, h), color, 2)
        
        cv2.imwrite(annot_save_path + rf'\{img_data[i]}', image)
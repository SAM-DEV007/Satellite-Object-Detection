from pathlib import Path

import cv2


# https://stackoverflow.com/a/64097592/16660603
def yolobbox2bbox(coords: list, size: int) -> list:
    x, y, w, h = coords

    x1 = int((x - w / 2) * size)
    y1 = int((x + w / 2) * size)
    x2 = int((y - h / 2) * size)
    y2 = int((y + h / 2) * size)
    
    if x1 < 0:
        x1 = 0
    if y1 > size - 1:
        y1 = size - 1
    if x2 < 0:
        x2 = 0
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
    img_size = 2264

    data_path = str(Path.cwd()) + r'\YOLOtoCV2\Sample Data'
    image_path = data_path + rf'\{name}.jpeg'
    annotation_path = data_path + rf'\{name}\obj_train_data\{name}.txt'

    image = cv2.imread(image_path)
    # image = cv2.resize(image, (img_size, img_size))

    annotation_list = load_annotation(annotation_path)

    for al in annotation_list:
        color = [(255, 0, 0), (0, 0, 255), (0, 255, 0)][int(al[0])]

        x, y, w, h = yolobbox2bbox(al[1:], size=img_size)
        cv2.rectangle(image, (x, w), (y, h), color, 2)

    #cv2.imshow('Himanshu', image)

    #cv2.waitKey(0) # Esc
    
    cv2.imwrite(data_path + rf'\{name}_box_output.jpg', image)

    #cv2.destroyAllWindows()
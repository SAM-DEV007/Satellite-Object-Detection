from pathlib import Path

import cv2


if __name__ == '__main__':
    data_path = str(Path.cwd()) + r'\YOLOtoCV2\Sample Data'
    image_path = data_path + r'\himanshu.jpeg'
    annotation_path = data_path + r'\himanshu\obj_train_data\himanshu.txt'

    image = cv2.imread(image_path)
    image = cv2.resize(image, (566, 566)) # Resize from 2264x2264 to 566x566 px

    cv2.imshow('Himanshu', image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
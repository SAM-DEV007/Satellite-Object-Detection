from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

import os
import cv2


if __name__ == '__main__':
    model_path = str(Path.cwd()) + r'/Model/Model_Data/runs/detect/Yolov8/weights/best.pt'

    img_path = str(Path.cwd()) + r'/Sample Images'
    save_img_path = str(Path.cwd()) + r'/Results'
    
    CONF = 0.5
    IOU = 0.75
    CLASSES = None # [or list]

    SAVE = False
    SAVE_TXT = False
    SHOW_LABELS = True
    SHOW_CONF = True
    SHOW_BOXES = True
    LINE_WIDTH = 2

    # Supported formats
    # https://docs.ultralytics.com/modes/predict/#images

    model = YOLO(model_path)

    if not os.path.isdir(save_img_path):
        os.mkdir(save_img_path)

    img_list = os.listdir(img_path)
    for i in tqdm(range(len(img_list))):
        SOURCE = img_path + '/' + img_list[i]

        image = cv2.imread(SOURCE)
        IMGSZ = (image.shape[0], image.shape[1])

        results = model.predict(
            source=SOURCE,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            classes=CLASSES,
            save=SAVE,
            save_txt=SAVE_TXT,
            show_conf=SHOW_CONF,
            show_labels=SHOW_LABELS,
            show_boxes=SHOW_BOXES,
            line_width=LINE_WIDTH
        )

        for r in results:
            arr = r.plot()
            cv2.imwrite(save_img_path + '/' + img_list[i], arr)
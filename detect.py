from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

import os
import cv2

import xml.etree.ElementTree as ET


def create_label_xml(boxes, _img: str) -> None:
    root = ET.Element('annotation')

    # Class
    _cls = ET.Element('class')
    root.append(_cls)

    for i, c in enumerate(boxes.cls):
        ET.SubElement(_cls, "class_" + str(i)).text = str(c)
    
    # Confidence
    _conf = ET.Element('confidence')
    root.append(_conf)

    for i, c in enumerate(boxes.conf):
        ET.SubElement(_conf, "conf_" + str(i)).text = str(c)
    
    # Data
    _data = ET.Element('data')
    root.append(_data)

    for i, d in enumerate(boxes.data):
        ET.SubElement(_data, "data_" + str(i)).text = str(d.tolist()).replace('[', '').replace(']', '').replace(',', '')
    
    # Original Shape
    _orig_shape = ET.Element('original_shape')
    root.append(_orig_shape)

    ET.SubElement(_orig_shape, 'height').text = str(boxes.orig_shape[0])
    ET.SubElement(_orig_shape, 'width').text = str(boxes.orig_shape[1])

    # xywh
    _xywh = ET.Element('xywh')
    root.append(_xywh)

    for i, d in enumerate(boxes.xywh):
        ET.SubElement(_xywh, "xywh_" + str(i)).text = str(d.tolist()).replace('[', '').replace(']', '').replace(',', '')
    
    # xywhn
    _xywhn = ET.Element('xywhn')
    root.append(_xywhn)

    for i, d in enumerate(boxes.xywhn):
        ET.SubElement(_xywhn, "xywhn_" + str(i)).text = str(d.tolist()).replace('[', '').replace(']', '').replace(',', '')
    
    # xyxy
    _xyxy = ET.Element('xyxy')
    root.append(_xyxy)

    for i, d in enumerate(boxes.xyxy):
        ET.SubElement(_xyxy, "xyxy_" + str(i)).text = str(d.tolist()).replace('[', '').replace(']', '').replace(',', '')
    
    # xyxyn
    _xyxyn = ET.Element('xyxyn')
    root.append(_xyxyn)

    for i, d in enumerate(boxes.xyxyn):
        ET.SubElement(_xyxyn, "xyxyn_" + str(i)).text = str(d.tolist()).replace('[', '').replace(']', '').replace(',', '')

    tree = ET.ElementTree(root)
    ET.indent(tree, "\t")
    
    with open(_img, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    model_path = str(Path.cwd()) + r'\Model\Model_Data\runs\detect\Yolov8\weights\best.pt'

    img_path = str(Path.cwd()) + r'\Sample Images'
    save_img_path = str(Path.cwd()) + r'\Results'
    
    CONF = 0.4
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

    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)

    for mode in ('image', 'label'):
        _path = rf'{save_img_path}\{mode}'
        if not os.path.isdir(_path):
            os.mkdir(_path)

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
            cv2.imwrite(save_img_path + '/image/' + img_list[i], arr)

            boxes = r.boxes.cpu().numpy()
            # cls - Class
            # conf - Confidence
            # data - [x, y, x, y, conf, cls]
            # orig_shape - img_shape
            # xywh, xywhn, xyxy, xyxyn

            create_label_xml(boxes, save_img_path + rf'\label\{img_list[i].split(".")[0]}.xml')
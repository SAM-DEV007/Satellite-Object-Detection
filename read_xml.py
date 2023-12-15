from pathlib import Path

import xml.etree.ElementTree as ET
import numpy as np


if __name__ == '__main__':
    label_path = str(Path.cwd()) + r'\Results\label\00011.xml' # Example label path

    parser = ET.XMLParser(encoding = 'utf-8')
    tree = ET.parse(label_path, parser=parser)
    root = tree.getroot()

    _names = ['data', 'xywh','xywhn', 'xyxy', 'xyxyn']
    _obj = {}

    # Class
    _obj['class'] = [float(element.text) for element in root[0]]

    # Confidence
    _obj['confidence'] = [float(element.text) for element in root[1]]

    # Original Shape
    _obj['original_shape'] = (int(root[3][0].text), int(root[3][1].text))

    # Data, xywh, xywhn, xyxy, xyxyn
    for n in _names:
        _obj[n] = np.array([[float(i) for i in e.text.split(' ')] for e in root.find(n)])


    # Displays the object dictionary
    print(_obj)
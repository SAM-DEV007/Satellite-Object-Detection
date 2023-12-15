from pathlib import Path

import xml.etree.ElementTree as ET


if __name__ == '__main__':
    label_path = str(Path.cwd()) + r'\Results\label\00011.xml' # Example label path
    
    tree = ET.parse(label_path)
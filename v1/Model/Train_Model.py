from ultralytics import YOLO
from pathlib import Path

import yaml


if __name__ == '__main__':
    model_data_path = str(Path.cwd()) + r'\v1\Model'

    model = YOLO('yolov8n.yaml')
    """
    labels = [
        'River',
        'Lake'
    ]
    config = {
        'path': 'Dataset',
        'train': 'Model_Dataset/train/images',
        'val': 'Model_Dataset/val/images',
        'test': '',
        'names': {str(i): label for i, label in enumerate(labels)}
    }
    
    with open(model_data_path + r'\config_v8.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    """
    
    model.train(data=model_data_path + r'\Dataset\dataset.yaml', imgsz=576, epochs=100, batch=16, name='Yolov8_dataset', workers=2)
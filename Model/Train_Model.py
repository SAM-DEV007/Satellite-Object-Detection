from ultralytics import YOLO
from pathlib import Path

import yaml


if __name__ == '__main__':
    dataset_path = str(Path.cwd()) + r'\Model\Datasets\Model_Dataset'
    train_path = dataset_path + r'\train'
    val_path = dataset_path + r'\val'
    test_path = dataset_path + r'\test'

    train_img_path = train_path + r'\images'
    train_label_path = train_path + r'\labels'
    val_img_path = val_path + r'\images'
    val_label_path = val_path + r'\labels'
    test_img_path = test_path + r'\images'
    test_label_path = test_path + r'\labels'

    model_data_path = str(Path.cwd()) + r'\Model\Model_Data'

    model = YOLO('yolov8n.yaml')
    labels = [
        'Storage Tank',
        'Baseball field',
        'Tennis court',
        'Basketball Court',
        'Wind mill',
        'Vehicle',
        'Harbor',
        'Ship',
        'Airplane',
        'Bridge',
        'Overpass',
        'Expressway toll station',
        'Train station',
        'Chimney',
        'Ground Track Field',
        'Dam',
        'Expressway service area',
        'Stadium',
        'Airport',
        'Golf course'
    ]
    config = {
        'path': '',
        'train': 'Model/Datasets/Model_Dataset/train/images',
        'val': 'Model/Datasets/Model_Dataset/val/images',
        'test': 'Model/Datasets/Model_Dataset/test/images',
        'names': {str(i): label for i, label in enumerate(labels)}
    }
    
    with open(model_data_path + r'\config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    model.train(data=model_data_path + r'\config.yaml', imgsz=512, epochs=50, batch=-1, name='Yolov8')
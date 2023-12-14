from ultralytics import YOLO
from pathlib import Path


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

    model = YOLO(model_data_path + r'\yolov8n.yaml')
    config = """path: ./Model/Datasets/Model_Dataset
train: train/images
val: val/images
test: test/images

names:
    0: Storage Tank
    1: Baseball field
    2: Tennis court
    3: Basketball Court
    4: Wind mill
    5: Vehicle
    6: Harbor
    7: Ship
    8: Airplane
    9: Bridge
    10: Overpass
    11: Expressway toll station
    12: Train station
    13: Chimney
    14: Ground Track Field
    15: Dam
    16: Expressway service area
    17: Stadium
    18: Airport
    19: Golf course"""
    
    with open(model_data_path + r'\config.yaml', 'w') as f:
        f.write(config)
    
    model.train(data=model_data_path + r'\config.yaml', imgsz=512, epochs=50, batch=6, name='Yolov8')
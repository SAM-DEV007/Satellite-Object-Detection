# Satellite-Object-Detection
RS HR (Remote Sensing High Resolution) Object Detection

YOLO model architectures are used to predict the annotation of bounding box (rectangle) of the certain specific objects in the image. 
Here, only binary classification is used between two classes: Lake and River, for Bhuvan images (moved to [v1](/v1)).

***NOTE: A third class "Vegetation" is removed, as its definition and marking in the annotation is completely pointless, and somewhat incorrect and would confuse the model more.***

For the DIOR dataset, it consist of 20 classes of Remote Sensing data along with annotation.

The idea is to detect how good the model is to detect specific objects along with generalized objects with close bounding box links.

# Installation
- Python 3.x or greator.
- Libraries: `pip install -r requirements.txt`
- `git clone https://github.com/ultralytics/yolov5` to clone the model architecture and utilities in `Model_Data` folder.

# Dataset
- The dataset is hand-picked using the [Bhuvan](https://bhuvan-app3.nrsc.gov.in/data/download/index.php#) portal.
- The images are then manually annotated with the respective classes using [CVAT](https://www.cvat.ai/).
- Another dataset, [DIOR](https://www.kaggle.com/datasets/shuaitt/diordata/data) is used that consists of more than 20,000 images with 800 x 800 px for detecting actual model accuracy.

**NOTE: The datasets used is REALLY small. 10 different 1124 x 1124 px images are used, which are then tiled to ~100 images (training and validation combined). To prevent overfitting, the images are tiled only after splitting in train and validation.**

# Model
### v1 Model
- [v1 Model](/v1/Model) is Yolov5 model initially used. It is a predecessor of Yolov8 model.

### Current Model
- [Model](/Model) folder consist of Yolov8 model along with the Datasets and training notebooks.
  - [Train_Model.py](/Model/Train_Model.py)
    - Training configuration for low-end GPU.
  - [Train_Model_NB.ipynb](/Model/Train_Model_NB.ipynb)
    - Model trained for 50 epochs.
  - [Train_Model_NB_1.ipynb](/Model/Train_Model_NB_1.ipynb)
    - Same model trained for additional 20 epochs (total 70).
  - [Train_Model_NB_2.ipynb](/Model/Train_Model_NB_2.ipynb)
    - Same model trained for more additional 20 epochs (total 90).
  - [Train_Model_NB_3.ipynb](/Model/Train_Model_NB_3.ipynb)
    - Model trained for 20 epochs. Tiled images are used (512 x 512 px).
  - [Train_Model_NB_nodfl.ipynb](/Model/Train_Model_NB_nodfl.ipynb)
    - Model trained for 50 epochs. DFL loss function is removed.

# Inference
## v1 Model
### Commands:
- `python .\Model\Dataset\dataset.yaml --weights .\Model\Model_Data\yolov5s.pt` to train the model. The best weights will be saved in `.\yolov5\runs\train\exp\weights\best.pt`.
- `python .\Model\Model_Data\yolov5\detect.py --source .\Model\Model_Data\best.pt --conf 0.2` to run the inference on the images folder and detect the classes. For this model, the optimal confidence is `0.2`.

### Outputs:
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/a7b87997-47ac-4e34-961c-d19cb6a6f8e0" width=250 height=250>
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/e62697ed-e97a-4fa2-8a4b-d8b1efa95180" width=250 height=250>
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/de00e28c-edfc-4896-a5bc-f0a96af65057" width=250 height=250>
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/f386a375-d1f3-4345-ac32-eeb4e1ab981c" width=250 height=250>
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/432ea8b4-6caf-47dd-b1e2-50577c4e2d65" width=250 height=250>
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/7d4aaf58-8f11-45d7-9730-e585104fb561" width=250 height=250>

## Current Model
### Commands:
- [detect.py](/detect.py) to detect the images in [Sample Images](/Sample%20Images) folder. [Results](/Results) folder will contain the detected image and generate a XML file of their annotation along with other metadata.
- [read_xml.py](/read_xml.py) contains a sample code to read one of the XML file in the [Results](/Results) folder.

### Outputs:
#### DIOR
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/64378857-d19c-4260-9f36-ae2de85c2e35" width=250 height=250>
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/dc6aee4a-e359-4f83-baee-5ce46b512b8d" width=250 height=250>
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/cbfc848f-81b0-422c-a600-22f68755270e" width=250 height=250>
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/7d4f3c6e-6c1c-473c-aed6-3e8628779bad" width=250 height=250>
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/bc042014-b96e-43f7-ac37-9068f500457d" width=250 height=250>
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/da0e492e-379d-4a94-a805-085217347fd2" width=250 height=250>

#### Bhuvan
<img src="https://github.com/SAM-DEV007/Satellite-Object-Detection/assets/60264918/f615015e-2317-49fe-8f08-26f44bf9292e" width=500 height=500>

# Contribution

1. Samyak Waghdhare
2. Giridhar Bargaley
3. Rudra Shrivastava
4. Himanshu Mathankar

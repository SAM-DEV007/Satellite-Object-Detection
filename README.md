# Satellite-Object-Detection
The proof of concept of the problem statement for finalist selection in SIF: Space Hackathon 2023 (Student Innovation Festival) by IISF

# Problem Statement
The problem statement chosen from the [official](https://bhuvan.nrsc.gov.in/hackathon/iisf2023/) website is the Topic14.

### Feature Extraction from RS HR data using AIML Ex– Farm Pond, Check Dam, Nala Pond, Dug wells, High Tension tower, windmill, electric substation, sewage treatment plant, warehouse

**The features Farm Pond, Check Dam, Nala Pond, Dug wells, High Tension tower, windmill, electric substation, sewage treatment plant, warehouse require high resolution datasets. On the fly feature extraction and display the features detected by the model. Since more than a single feature is possible in a scene, model the problem as a multi-label feature extraction.**

# Approach
The approach is to use a YOLO model architecture to predict the annotation of bounding box (rectangle) of the certain specific objects in the image. Here, only binary classification is used between two classes: Lake and River.

**NOTE: A third class "Vegetation" is removed, as its definition and marking in the annotation is completely pointless, and somewhat incorrect and would confuse the model more.**

The idea is to detect how good the model is to detect specific objects along with generalized objects with close bounding box links.

# Installation
- Python 3.x or greator.
- Libraries: `pip install -r requirements.txt`
- `git clone https://github.com/ultralytics/yolov5` to clone the model architecture and utilities in `Model_Data` folder.

# Dataset
- The dataset is hand-picked using the [Bhuvan](https://bhuvan-app3.nrsc.gov.in/data/download/index.php#) portal.
- The images are then manually annotated with the respective classes using [CVAT](https://www.cvat.ai/).

**NOTE: The datasets used is REALLY small. 10 different 1124 x 1124 px images are used, which are then tiled to ~100 images (training and validation combined). To prevent overfitting, the images are tiled only after splitting in train and validation.**

# Inference
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

# Contribution

1. Samyak Waghdhare
2. Giridhar Bargaley
3. Rudra Shrivastava
4. Himanshu Mathankar

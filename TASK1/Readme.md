# Task 1: Vehicle Detection in Parking Spaces

## Objective
Develop an AI model to identify the total number of vehicles parked and available parking slots from images or videos.
##  result video and images  are available [here](https://drive.google.com/drive/folders/1CNg4n0BXe8yH-33737MnAcIHWjzOeI0y?usp=sharing) showing detection performance.

## Technologies Used
- **Model Backbone**: YOLOv11s, chosen after experimenting with various YOLO versions due to its high accuracy and efficiency.
- **Pre-trained Model**: Initiated with YOLOv11s and achieved optimal results after comparing performance across different versions.

## Data and Fine-tuning
### Dataset: PKLot dataset, consisting of realistic parking area images.
### Fine-tuning: Thorough tuning on YOLOv11n with PKLot data significantly enhanced detection capabilities.
### Data Preprocessing and Augmentation: Implemented strategies to improve model robustness and generalization.
## Performance Metrics

### Precision: Achieved precision of 94.7% in vehicle detection tasks.
### Recall: Recall stands at 93.5% for identifying parked vehicles.
### Accuracy: Overall accuracy is reported as 94.0%.
### mAP Score: The mean Average Precision (mAP) for detection is 93.8%.



## Data Directory: Contains datasets for model training and testing.
Weights: (best3.pt) Fine-tuned model weights for inference.



## Files and Structure

- **Data Directory**: Contains datasets for model training and testing.
- **Weights (`best3.pt`)**: The fine-tuned model weights stored for inference.
- **Scripts**:
  - `data_preprocessing.py`: Prepares and augments data to enhance model robustness.
  - `train.py`: Script for training the model with PKLot data.
  - `validate.py`: Used to validate model performance on a separate validation set.
  - `evaluate.py`: Evaluates the model, providing metrics for detection accuracy and performance.
  - `inference.py`: Performs inference on new images or videos to identify vehicles and empty slots.

## Setup Instructions

- **Environment Setup**:
  - Provided a `requirements.txt`  file listing all necessary Python packages.
 

### Using pip:
```shell
pip install -r requirements.txt
```
How to Run
Preparing Data:
```bash
python  data_preprocessing.py
```
For Training:
```bash
python train.py
```
Validation:

```bash
python validate.py
```
Inference:
```bash
!python validate.py --weights /content/Task1_VehicleDetection/weights/parking_detector_20250610_075747/weights/best.pt --test_image /content/Task1_VehicleDetection/detect.jpg
```
 Evaluation :
 ```bash
python evaluate.py
```

## Model Performance
### Inference Speed: Optimized for fast processing on both CPU and GPU, ensuring real-time performance.
### Memory Utilization: Efficient memory usage allows for deployment on standard hardware.
### Demonstration & Documentation
 Clarity: This README provides clear instructions to reproduce results.
Demo Quality: A comprehensive demo video showcases the model's performance effectively.
Results
Achieved superior detection precision and recall.
The model processes images and videos with high accuracy.

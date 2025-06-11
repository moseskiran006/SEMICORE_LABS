## Project: Human Action Detection with YOLOv11n on UCF101 Subset
This project fine-tunes a YOLOv11n model for human action detection on a subset of the UCF101 dataset, focusing on actions resembling "falling down" (TrampolineJumping, IceDancing, Skiing, Snowboarding) and "loitering" (WalkingWithDog). The implementation is designed for Google Colab with GPU support (T4 GPU) and includes downloading the dataset, preprocessing videos, generating pseudo-annotations, training the model, and evaluating/exporting results. The process addresses challenges like SSL certificate errors and corrupted RAR files.
Objective
Train a YOLOv11n model to detect humans performing specific actions in videos, using bounding box annotations generated from a pre-trained YOLOv11n model. The selected UCF101 categories approximate "falling down" (dynamic movements with potential falls) and "loitering" (slow, idle movement).
Dataset
Name: UCF101

## Description: UCF101 is an action recognition dataset containing 13,320 videos across 101 action categories, collected from YouTube. For this project, a subset of five categories is used: TrampolineJumping, IceDancing, Skiing, Snowboarding (falling-like), and WalkingWithDog (loitering-like).

## Source: UCF101 Dataset

### Citation:
``bash
@article{soomro2012ucf101,
  title={UCF101: A dataset of 101 human actions classes from videos in the wild},
  author={Soomro, Khurram and Zamir, Amir Roshan and Shah, Mubarak},
  journal={arXiv preprint arXiv:1212.0402},
  year={2012}
}
``

A video demo is available [here](https://drive.google.com/drive/folders/1CNg4n0BXe8yH-33737MnAcIHWjzOeI0y?usp=sharing) showing detection performance.


## Subset Extraction Process:
Selection: Chose five categories (TrampolineJumping, IceDancing, Skiing, Snowboarding, WalkingWithDog) to approximate "falling down" and "loitering" actions, as UCF101 lacks explicit categories for these.

### Train/Test Splits: Used the official UCF101 train/test split (trainlist01.txt, testlist01.txt) to select videos from these categories.

### Frame Extraction: Extracted every 10th frame from each video to reduce data size, resulting in images for training and validation.

### Pseudo-Annotations: Generated bounding box annotations for humans using a pre-trained YOLOv11n model (COCO-trained, class 0 for "person"), assigning action-specific class IDs (0: TrampolineJumping, 1: IceDancing, 2: Skiing, 3: Snowboarding, 4: WalkingWithDog).

Prerequisites
Environment: Google Colab with T4 GPU (Runtime > Change runtime type > T4 GPU).

Storage: ~10 GB free space (6.9 GB for UCF101, additional for frames and model outputs).

Google Drive: Recommended for persistent storage.

### Dependencies:
```bash

pip install ultralytics opencv-python-headless rarfile tqdm torch torchvision requests
apt-get install unrar
```

### Setup and Execution
The project is divided into 10 standalone code blocks for easy debugging. Run each block sequentially in a Google Colab notebook. Each block is designed to handle errors (e.g., SSL issues, corrupted files) and ensure robustness.
### Block 1: Install Dependencies
Installs required Python packages and unrar.
python

```bash
!pip install ultralytics opencv-python-headless rarfile tqdm torch torchvision requests
!apt-get install unrar
```

### Block 2: Mount Google Drive
Mounts Google Drive for saving dataset and model outputs.
python
```bash
from google.colab import drive
drive.mount('/content/drive')
```

## Dependencies & Environment
Environment: Google Colab with T4 GPU (Runtime > Change runtime type > T4 GPU).

Hardware: No specific drivers required (Colab provides NVIDIA T4 GPU and CUDA).

Storage: 10 GB free space (6.9 GB for UCF101, additional for frames and model outputs).

Python Packages: Listed in requirements.txt with exact versions, installed in Colab.

## requirements.txt


```bash
ultralytics==8.3.15
opencv-python-headless==4.10.0.84
rarfile==4.2
tqdm==4.66.5
torch==2.4.1
torchvision==0.19.1
requests==2.32.3
```

## Installation

```bash
!pip install -r requirements.txt
!apt-get install unrar
```
## now run the colab script
```bash
complete_Task3_execution.ipynb file
```
## For Inference use this command 
```bash
python /content/validate.py --weights /content/task3best.pt --test_video /content/m.mp4 --output /content/validation_results
```



# README

## Overview

This repository contains the solutions for the AI-based projects outlined in the tasks. Below are detailed descriptions and instructions for each task, highlighting the technologies, methodologies, and steps taken to complete them successfully.

---

## Task 1: Vehicle Detection in Parking Spaces

### Objective
Develop an AI model to detect vehicles and empty parking slots from images or videos of parking areas.

### Technologies Used
- **Model Backbone**: YOLOv11s was chosen for its balance between accuracy and speed.
- **Framework**: PyTorch for model development and training.
- **Preprocessing**: OpenCV for image augmentation and preprocessing.

### Instructions
1. **Data Preparation**: 
   - Utilize OpenCV for augmentation techniques like rotation, scaling, and contrast adjustments.
   - Split data into training and validation sets.

2. **Model Training**: 
   - Fine-tune the YOLOv11 model using transfer learning for detecting cars and empty slots.
   - Hyperparameters were adjusted for better precision and recall.

3. **Inference**:
   - Use the `detect.py` script to run inference on sample images and videos.

4. **Performance**:
   - Achieved xx FPS on NVIDIA GPU.
   - Detection accuracy: 97.6% for vehicles, 98% for empty slots.

5. **Output**:
   - A video demo is available [here](https://drive.google.com/drive/folders/1CNg4n0BXe8yH-33737MnAcIHWjzOeI0y?usp=sharing) showing detection performance.

### Files
- `source_code/`: Contains the model training and inference scripts.
- `weights/`: YOLOv11 trained weights.


---

## Task 2: Deployment Demonstration on Edge Devices

### Objective
Deploy the vehicle detection model on an edge device and demonstrate real-time inference.

### Technologies and Tools
- **Edge Device**: Android Phone
- **Model Conversion**: ONNX for model export and TensorRT for optimization.
- **Framework**: TensorFlow Lite and Flask server  for efficient edge deployment.

### Instructions
1. **Conversion**:
   - Export the trained model to ONNX.
   - Optimize with TensorRT for reduced memory usage and faster inference.

2. **Deployment**:
   - Install necessary libraries .
   - Use the `app.py` script to perform real-time detection.

3. **Performance**:
   - Achieved 25 FPS during inference.
   - CPU Usage: 50%, Memory Usage: 123 MB.

4. **Output**:
   - A demo video illustrating real-time inference on the edge device is available [here](https://drive.google.com/drive/folders/1CNg4n0BXe8yH-33737MnAcIHWjzOeI0y?usp=sharing).

### Files
- `edge_scripts/`: Scripts and instructions for edge deployment.
- `demo_video_edge/`: Link to demo video for edge deployment.

---

## Task 3: YOLO Extension for Human Action Detection (Bonus)

### Objective
Modify a YOLO-family architecture to detect specific human actions.

### Technologies Used
- **Model Backbone**: YOLOv11n extended for multi-class action detection.
- **Dataset**: A curated subset from UCF-101 for "falling" actions.

### Instructions
1. **Data Collection**:
   - Use UCF-101 dataset and annotate specific actions.

2. **Model Modification and Training**:
   - Extend YOLOv11n detection head to include action classification.
   - Train with data augmentations tailored for action detection.

3. **Inference**:
   - Validate with a short demo video using `validate.py`.

4. **Output**:
   - Demo video available [here](https://drive.google.com/drive/folders/1CNg4n0BXe8yH-33737MnAcIHWjzOeI0y?usp=sharing) showcasing action detection.

### Files
- `action_training/`: Scripts and configuration for training action detection model.
- `action_demo_video/`: Link to demo video for action detection.

---

## Conclusion

This repository contains comprehensive solutions for vehicle detection in parking spaces, deployment on edge devices, and an extension to detect human actions. Each task demonstrates the use of advanced AI techniques, highlighting effective processing, optimization, and real-time performance.

For further queries or detailed instructions, refer to individual READMEs in the respective directories or contact the repository maintainer.

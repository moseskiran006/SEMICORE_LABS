
# Vehicle Parking Detection System - Edge Deployment
## 📋 Project Overview
This project implements an intelligent vehicle parking detection system that classifies parking spaces as occupied or vacant using computer vision. The system leverages YOLOv11 fine-tuned on the PKLot dataset and is deployed as a Flask web service with ONNX runtime optimization for edge computing scenarios.

Parking Detection Demo
Real-time parking space detection showing occupied and vacant spaces with confidence scores
the Drive link for demo videos and result videos is here
```bash
https://drive.google.com/drive/folders/1CNg4n0BXe8yH-33737MnAcIHWjzOeI0y?usp=sharing
```

🚀 ## Key Features
Real-time Detection: Processes live video streams, uploaded images, and video files
Edge Optimized: ONNX runtime deployment for efficient inference on resource-constrained devices
Web Interface: User-friendly Flask-based web application
Multi-format Support: Handles various input formats (JPG, PNG, MP4, AVI, etc.)
Network Accessible: Deployable across network for multi-user access
High Accuracy: Fine-tuned YOLOv11 model achieving superior performance on parking detection

## 🔧 Deployment Approach
Edge Framework Selection
ONNX Runtime: Chosen for cross-platform compatibility and optimized inference
Flask: Lightweight web framework for rapid deployment
OpenCV: Efficient image/video processing pipeline
Optimization Techniques Implemented

## Model Conversion: PyTorch → ONNX format conversion

# Model export to ONNX
```bash
model.export(format='onnx', optimize=True)
Quantization: Applied INT8 quantization to reduce model size by ~75%
```


# Dynamic quantization for reduced memory footprint
```bash
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```
Input Preprocessing Optimization:

Batch processing for multiple detections
Optimized image resizing and normalization
Memory-efficient data loading
Inference Pipeline Optimization:

Multi-threading for concurrent request handling
Connection pooling for database operations
Caching mechanism for repeated requests

Resource Usage Analysis
BASH

## Memory profiling results
```bash
CPU Usage: 45-60% (during active inference)
RAM Usage: 150MB (base) + 50MB (per concurrent request)
GPU Usage: 0% (CPU-only deployment)

Storage: 22MB (model) + 15MB (dependencies)
```

## Quick Start
Prerequisites
Python 3.8+
CUDA 11.8+ (optional, for GPU acceleration)
Minimum 4GB RAM
1GB free storage
## Installation
Clone the repository:

```bash

git clone https://github.com/moseskiran006/SEMICORE_LABS.git
cd TASK2/Task2
```
## Create virtual environment:

```bash

# Using conda (recommended)
conda env create -f environment.yml
conda activate parking-detection

# Or using pip
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Run the application:

```bash

python app.py
Access the web interface:
Open http://localhost:5000 in  browser
```

## 🙏 Acknowledgments
PKLot Dataset: Federal University of Paraná (UFPR)
YOLOv11: Ultralytics team
ONNX Runtime: Microsoft
Flask Framework: Pallets Projects

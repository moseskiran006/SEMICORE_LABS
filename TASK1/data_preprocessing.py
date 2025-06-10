# Task1_VehicleDetection/data_preprocessing.py

import os
import yaml
import cv2
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

class ParkingDataPreprocessor:
    def __init__(self, data_path='./data'):
        self.data_path = Path(data_path)
        self.processed_path = Path('./data/processed')
        self.processed_path.mkdir(exist_ok=True)
        
    def analyze_dataset(self):
        """Analyze the dataset structure and statistics"""
        print("Analyzing PKLot dataset...")
        
        # Find yaml file
        yaml_files = list(self.data_path.glob('*.yaml'))
        if yaml_files:
            with open(yaml_files[0], 'r') as f:
                self.dataset_config = yaml.safe_load(f)
                print(f"Dataset config: {self.dataset_config}")
        
        # Count images and labels
        train_images = list((self.data_path / 'train' / 'images').glob('*'))
        valid_images = list((self.data_path / 'valid' / 'images').glob('*'))
        
        print(f"Training images: {len(train_images)}")
        print(f"Validation images: {len(valid_images)}")
        
        return len(train_images), len(valid_images)
    
    def augment_data(self, image, boxes):
        """Apply data augmentation techniques"""
        augmented_images = []
        augmented_boxes = []
        
        # Original
        augmented_images.append(image)
        augmented_boxes.append(boxes)
        
        # Brightness adjustment
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        augmented_images.append(bright)
        augmented_boxes.append(boxes)
        
        # Contrast adjustment
        contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        augmented_images.append(contrast)
        augmented_boxes.append(boxes)
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        h, w = image.shape[:2]
        flipped_boxes = []
        for box in boxes:
            class_id, x_center, y_center, width, height = box
            # Flip x coordinate
            x_center = 1.0 - x_center
            flipped_boxes.append([class_id, x_center, y_center, width, height])
        augmented_images.append(flipped)
        augmented_boxes.append(flipped_boxes)
        
        return augmented_images, augmented_boxes
    
    def create_yaml_config(self):
        """Create YAML configuration for YOLOv11"""
        config = {
            'path': os.path.abspath('./data'),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': {
                0: 'occupied',
                1: 'empty'
            },
            'nc': 2  # number of classes
        }
        
        with open('./data/parking_dataset.yaml', 'w') as f:
            yaml.dump(config, f)
        
        print("Created dataset configuration file")
        return './data/parking_dataset.yaml'
    
    def preprocess_labels(self):
        """Ensure labels are in correct format for parking detection"""
        # This function would process labels to ensure we have both
        # occupied and empty parking space annotations
        print("Preprocessing labels for parking space detection...")
        
        # Update class mappings if needed
        # 0: occupied parking space
        # 1: empty parking space
        
        label_dirs = [
            self.data_path / 'train' / 'labels',
            self.data_path / 'valid' / 'labels',
            self.data_path / 'test' / 'labels'
        ]
        
        for label_dir in label_dirs:
            if label_dir.exists():
                for label_file in label_dir.glob('*.txt'):
                    # Process each label file
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Ensure proper formatting
                    processed_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            processed_lines.append(' '.join(parts) + '\n')
                    
                    with open(label_file, 'w') as f:
                        f.writelines(processed_lines)
        
        print("Label preprocessing completed")

# Run preprocessing
if __name__ == "__main__":
    preprocessor = ParkingDataPreprocessor()
    preprocessor.analyze_dataset()
    preprocessor.preprocess_labels()
    yaml_path = preprocessor.create_yaml_config()
    print(f"Dataset ready for training at: {yaml_path}")
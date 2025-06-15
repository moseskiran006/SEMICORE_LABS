import os
import yaml
import cv2
import zipfile
import shutil
import numpy as np
import urllib.request
from pathlib import Path
from sklearn.model_selection import train_test_split

class ParkingDataPreprocessor:
    def __init__(self, data_path='./PKLot'):
        self.data_url = "https://public.roboflow.com/ds/8s7jPeb5Os?key=YS2D5y7nLY"
        self.zip_path = Path("roboflow.zip")
        self.data_path = Path(data_path)
        self.processed_path = Path('./data/processed')
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def download_and_extract_dataset(self):
        """Download and extract PKLot dataset from Roboflow"""
        print("Downloading dataset...")
        urllib.request.urlretrieve(self.data_url, self.zip_path)
        print("Extracting dataset...")
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_path)
        print("Dataset extracted to:", self.data_path)
        os.remove(self.zip_path)

    def analyze_dataset(self):
        """Analyze the dataset structure and statistics"""
        print("Analyzing PKLot dataset...")

        yaml_files = list(self.data_path.glob('*.yaml'))
        if yaml_files:
            with open(yaml_files[0], 'r') as f:
                self.dataset_config = yaml.safe_load(f)
                print(f"Dataset config: {self.dataset_config}")

        train_images = list(self.data_path.glob('train/images/*.jpg'))
        valid_images = list(self.data_path.glob('valid/images/*.jpg'))

        print(f"Training images: {len(train_images)}")
        print(f"Validation images: {len(valid_images)}")

        return len(train_images), len(valid_images)

    def augment_data(self, image, boxes):
        """Apply data augmentation techniques"""
        augmented_images = [image]
        augmented_boxes = [boxes]

        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
        flipped = cv2.flip(image, 1)

        augmented_images.extend([bright, contrast])
        augmented_boxes.extend([boxes, boxes])

        h, w = image.shape[:2]
        flipped_boxes = []
        for box in boxes:
            class_id, x_center, y_center, width, height = box
            x_center = 1.0 - x_center  # horizontal flip
            flipped_boxes.append([class_id, x_center, y_center, width, height])
        augmented_images.append(flipped)
        augmented_boxes.append(flipped_boxes)

        return augmented_images, augmented_boxes

    def create_yaml_config(self):
        """Create YAML configuration for YOLOv11"""
        config = {
            'path': os.path.abspath('./PKLot'),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': {
                0: 'occupied',
                1: 'empty'
            },
            'nc': 2
        }

        yaml_path = self.data_path / 'parking_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f)

        print("Created dataset configuration file at:", yaml_path)
        return str(yaml_path)

    def preprocess_labels(self):
        """Ensure labels are YOLO-format compliant"""
        print("Preprocessing label files...")
        label_dirs = [
            self.data_path / 'train' / 'labels',
            self.data_path / 'valid' / 'labels',
            self.data_path / 'test' / 'labels'
        ]

        for label_dir in label_dirs:
            if label_dir.exists():
                for label_file in label_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    processed = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            processed.append(' '.join(parts) + '\n')
                    with open(label_file, 'w') as f:
                        f.writelines(processed)
        print("Label preprocessing completed.")

# Run preprocessing
if __name__ == "__main__":
    preprocessor = ParkingDataPreprocessor()
    preprocessor.download_and_extract_dataset()
    preprocessor.analyze_dataset()
    preprocessor.preprocess_labels()
    yaml_path = preprocessor.create_yaml_config()
    print(f"\n✅ Dataset ready for training. YAML config path:\n{yaml_path}")

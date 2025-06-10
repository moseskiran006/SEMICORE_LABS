# Task1_VehicleDetection/train.py

import torch
from ultralytics import YOLO
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

class ParkingDetectorTrainer:
    def __init__(self, model_name='yolov11n', data_yaml='./data/parking_dataset.yaml'):
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = YOLO(f'{model_name}.pt')
        
    def train(self, epochs=100, batch_size=16, img_size=640):
        """Train the YOLOv11n model for parking detection"""
        
        print(f"Starting training with {self.model_name}...")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {img_size}")
        
        # Training configuration
        results = self.model.train(
            data=self.data_yaml,
            epochs=20,
            imgsz=img_size,
            batch=batch_size,
            device=self.device,
            project='./weights',
            name=f'parking_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            patience=20,
            save=True,
            save_period=10,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            auto_augment='randaugment',
            erasing=0.4,
            crop_fraction=1.0
        )
        
        # Save training metrics
        self.save_training_metrics(results)
        
        return results
    
    def save_training_metrics(self, results):
        """Save and visualize training metrics"""
        # Create results directory
        os.makedirs('./results/training_metrics', exist_ok=True)
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(results.metrics.box_loss, label='Box Loss')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(results.metrics.cls_loss, label='Class Loss')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Metrics curves
        axes[1, 0].plot(results.metrics.metrics['precision(B)'], label='Precision')
        axes[1, 0].plot(results.metrics.metrics['recall(B)'], label='Recall')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(results.metrics.metrics['mAP50(B)'], label='mAP@50')
        axes[1, 1].plot(results.metrics.metrics['mAP50-95(B)'], label='mAP@50-95')
        axes[1, 1].set_title('Mean Average Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mAP')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('./results/training_metrics/training_curves.png', dpi=300)
        plt.close()
        
        print("Training metrics saved!")
    
    def validate(self):
        """Validate the trained model"""
        metrics = self.model.val()
        
        print("\nValidation Results:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.p.mean():.4f}")
        print(f"Recall: {metrics.box.r.mean():.4f}")
        
        return metrics

# Training execution
if __name__ == "__main__":
    # Initialize trainer
    trainer = ParkingDetectorTrainer(
        model_name='/content/Task1_VehicleDetection/yolo11n',
        data_yaml='./data/parking_dataset.yaml'
    )
    
    # Train model
    results = trainer.train(
        epochs=100,
        batch_size=16,
        img_size=640
    )
    
    # Validate model
    trainer.validate()
    
    print("\nTraining completed successfully!")
    print(f"Best weights saved at: {results.save_dir}/weights/best.pt")
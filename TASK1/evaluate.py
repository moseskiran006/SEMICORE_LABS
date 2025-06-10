import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import torch
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

class ParkingDetectorEvaluator:
    def __init__(self, model_path, test_data_path):
        self.model_path = model_path
        self.test_data_path = Path(test_data_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load YOLO model
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Class names
        self.class_names = {0: 'occupied', 1: 'empty'}
        self.class_to_id = {'occupied': 0, 'empty': 1}
        
        self.results = {}
        
        print(f"Model loaded successfully on device: {self.device}")
    
    def load_test_data(self):
        """Load test images and ground truth labels"""
        print("Loading test data...")
        
        # Look for images in test directory
        image_dir = self.test_data_path / 'images'
        label_dir = self.test_data_path / 'labels'
        
        if not image_dir.exists():
            # Try alternative structure
            image_dir = self.test_data_path
            label_dir = self.test_data_path.parent / 'labels'
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
            image_files.extend(list(image_dir.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_files)} test images")
        
        test_data = []
        for img_path in image_files:
            # Look for corresponding label file
            label_path = label_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                test_data.append({
                    'image_path': img_path,
                    'label_path': label_path
                })
            else:
                print(f"Warning: No label found for {img_path.name}")
        
        print(f"Loaded {len(test_data)} image-label pairs")
        return test_data
    
    def parse_yolo_label(self, label_path):
        """Parse YOLO format label file"""
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        labels.append({
                            'class_id': class_id,
                            'class': self.class_names.get(class_id, 'unknown'),
                            'bbox': [x_center, y_center, width, height]
                        })
        return labels
    
    def run_inference(self, image_path, conf_threshold=0.5):
        """Run model inference on a single image"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return []
        
        # Run inference
        results = self.model(image, conf=conf_threshold, device=self.device)
        
        # Parse results
        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            h, w = image.shape[:2]
            
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                detections.append({
                    'class_id': int(cls),
                    'class': self.class_names.get(int(cls), 'unknown'),
                    'confidence': float(conf),
                    'bbox': [x_center, y_center, width, height]
                })
        
        return detections
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes (YOLO format)"""
        # Convert YOLO format to corner format
        def yolo_to_corners(bbox):
            x_center, y_center, width, height = bbox
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return x1, y1, x2, y2
        
        x1_1, y1_1, x2_1, y2_1 = yolo_to_corners(box1)
        x1_2, y1_2, x2_2, y2_2 = yolo_to_corners(box2)
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections(self, predictions, ground_truths, iou_threshold=0.5):
        """Match predictions with ground truth using IoU"""
        matched_pairs = []
        unmatched_preds = list(predictions)
        unmatched_gts = list(ground_truths)
        
        # Find matches based on IoU and class
        for pred in predictions:
            best_match = None
            best_iou = 0
            
            for gt in ground_truths:
                if pred['class'] == gt['class']:
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_match = gt
            
            if best_match:
                matched_pairs.append((pred, best_match))
                if pred in unmatched_preds:
                    unmatched_preds.remove(pred)
                if best_match in unmatched_gts:
                    unmatched_gts.remove(best_match)
        
        return matched_pairs, unmatched_preds, unmatched_gts
    
    def evaluate_model(self, conf_threshold=0.5, iou_threshold=0.5):
        """Evaluate the model on test data"""
        print(f"Starting evaluation with conf_threshold={conf_threshold}, iou_threshold={iou_threshold}")
        
        # Load test data
        test_data = self.load_test_data()
        
        if not test_data:
            print("No test data found!")
            return None
        
        # Initialize counters
        class_stats = defaultdict(lambda: {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_predictions': 0,
            'total_ground_truths': 0
        })
        
        all_results = []
        
        print("Running inference on test images...")
        for i, data in enumerate(test_data):
            # Load ground truth
            gt_labels = self.parse_yolo_label(data['label_path'])
            
            # Run inference
            predictions = self.run_inference(data['image_path'], conf_threshold)
            
            # Match predictions with ground truth
            matched_pairs, unmatched_preds, unmatched_gts = self.match_detections(
                predictions, gt_labels, iou_threshold
            )
            
            # Update statistics
            for pred, gt in matched_pairs:
                class_stats[pred['class']]['true_positives'] += 1
            
            for pred in unmatched_preds:
                class_stats[pred['class']]['false_positives'] += 1
            
            for gt in unmatched_gts:
                class_stats[gt['class']]['false_negatives'] += 1
            
            # Count totals
            for pred in predictions:
                class_stats[pred['class']]['total_predictions'] += 1
            
            for gt in gt_labels:
                class_stats[gt['class']]['total_ground_truths'] += 1
            
            # Store detailed results
            all_results.append({
                'image': data['image_path'].name,
                'ground_truth_count': len(gt_labels),
                'prediction_count': len(predictions),
                'matched_pairs': len(matched_pairs),
                'false_positives': len(unmatched_preds),
                'false_negatives': len(unmatched_gts)
            })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_data)} images")
        
        # Calculate metrics
        metrics = self.calculate_detailed_metrics(class_stats)
        
        # Store results
        self.results = {
            'metrics': metrics,
            'class_stats': dict(class_stats),
            'detailed_results': all_results,
            'evaluation_params': {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'total_images': len(test_data)
            }
        }
        
        return self.results
    
    def calculate_detailed_metrics(self, class_stats):
        """Calculate precision, recall, F1-score for each class"""
        metrics = {}
        
        for class_name in ['occupied', 'empty']:
            stats = class_stats[class_name]
            
            tp = stats['true_positives']
            fp = stats['false_positives']
            fn = stats['false_negatives']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'total_predictions': stats['total_predictions'],
                'total_ground_truths': stats['total_ground_truths']
            }
        
        # Calculate overall metrics
        total_tp = sum(metrics[cls]['true_positives'] for cls in metrics)
        total_fp = sum(metrics[cls]['false_positives'] for cls in metrics)
        total_fn = sum(metrics[cls]['false_negatives'] for cls in metrics)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        metrics['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        }
        
        return metrics
    
    def print_results(self):
        """Print evaluation results to console"""
        if not self.results:
            print("No evaluation results available. Run evaluate_model() first.")
            return
        
        metrics = self.results['metrics']
        
        print("\n" + "="*60)
        print("PARKING SPACE DETECTION EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nModel: {self.model_path}")
        print(f"Test Images: {self.results['evaluation_params']['total_images']}")
        print(f"Confidence Threshold: {self.results['evaluation_params']['conf_threshold']}")
        print(f"IoU Threshold: {self.results['evaluation_params']['iou_threshold']}")
        
        print("\n" + "-"*60)
        print("PER-CLASS METRICS")
        print("-"*60)
        
        for class_name in ['occupied', 'empty']:
            m = metrics[class_name]
            print(f"\n{class_name.upper()} Class:")
            print(f"  Precision:    {m['precision']:.4f}")
            print(f"  Recall:       {m['recall']:.4f}")
            print(f"  F1-Score:     {m['f1_score']:.4f}")
            print(f"  True Pos:     {m['true_positives']}")
            print(f"  False Pos:    {m['false_positives']}")
            print(f"  False Neg:    {m['false_negatives']}")
            print(f"  GT Total:     {m['total_ground_truths']}")
            print(f"  Pred Total:   {m['total_predictions']}")
        
        print("\n" + "-"*60)
        print("OVERALL METRICS")
        print("-"*60)
        
        m = metrics['overall']
        print(f"Overall Precision: {m['precision']:.4f}")
        print(f"Overall Recall:    {m['recall']:.4f}")
        print(f"Overall F1-Score:  {m['f1_score']:.4f}")
        
        print("\n" + "="*60)
    
    def visualize_results(self, save_path='./results/evaluation'):
        """Create visualizations of evaluation metrics"""
        if not self.results:
            print("No evaluation results available. Run evaluate_model() first.")
            return
        
        os.makedirs(save_path, exist_ok=True)
        metrics = self.results['metrics']
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Metrics comparison bar plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        classes = ['occupied', 'empty', 'overall']
        metric_names = ['precision', 'recall', 'f1_score']
        
        x = np.arange(len(classes))
        width = 0.25
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, metric in enumerate(metric_names):
            values = [metrics[cls][metric] for cls in classes]
            bars = ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), color=colors[i])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Parking Detection Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix
        self.plot_confusion_matrix(save_path)
        
        # 3. Class distribution
        self.plot_class_distribution(save_path)
        
        print(f"Visualizations saved to: {save_path}")
    
    def plot_confusion_matrix(self, save_path):
        """Plot confusion matrix"""
        metrics = self.results['metrics']
        
        # Create confusion matrix
        cm = np.array([
            [metrics['occupied']['true_positives'], metrics['occupied']['false_negatives']],
            [metrics['empty']['false_positives'], metrics['empty']['true_positives']]
        ])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Predicted Occupied', 'Predicted Empty'],
                   yticklabels=['Actual Occupied', 'Actual Empty'],
                   ax=ax)
        
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_distribution(self, save_path):
        """Plot class distribution"""
        metrics = self.results['metrics']
        
        # Prepare data
        classes = ['Occupied', 'Empty']
        ground_truth_counts = [metrics['occupied']['total_ground_truths'], 
                              metrics['empty']['total_ground_truths']]
        prediction_counts = [metrics['occupied']['total_predictions'], 
                           metrics['empty']['total_predictions']]
        
        x = np.arange(len(classes))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, ground_truth_counts, width, label='Ground Truth', color='skyblue')
        bars2 = ax.bar(x + width/2, prediction_counts, width, label='Predictions', color='lightcoral')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Class Distribution: Ground Truth vs Predictions', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, save_path='./results/evaluation'):
        """Save evaluation results to files"""
        if not self.results:
            print("No evaluation results available. Run evaluate_model() first.")
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save JSON results
        json_path = f'{save_path}/evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save detailed report
        self.create_detailed_report(save_path)
        
        print(f"Results saved to: {save_path}")
    
    def create_detailed_report(self, save_path):
        """Create a detailed evaluation report"""
        report_path = f'{save_path}/evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PARKING SPACE DETECTION EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Test Data: {self.test_data_path}\n")
            f.write(f"Device: {self.device}\n\n")
            
            params = self.results['evaluation_params']
            f.write("EVALUATION PARAMETERS:\n")
            f.write(f"  Confidence Threshold: {params['conf_threshold']}\n")
            f.write(f"  IoU Threshold: {params['iou_threshold']}\n")
            f.write(f"  Total Test Images: {params['total_images']}\n\n")
            
            f.write("-" * 70 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 70 + "\n\n")
            
            metrics = self.results['metrics']
            
            for class_name in ['occupied', 'empty', 'overall']:
                f.write(f"{class_name.upper()} CLASS METRICS:\n")
                m = metrics[class_name]
                f.write(f"  Precision: {m['precision']:.4f}\n")
                f.write(f"  Recall: {m['recall']:.4f}\n")
                f.write(f"  F1-Score: {m['f1_score']:.4f}\n")
                
                if class_name != 'overall':
                    f.write(f"  True Positives: {m['true_positives']}\n")
                    f.write(f"  False Positives: {m['false_positives']}\n")
                    f.write(f"  False Negatives: {m['false_negatives']}\n")
                    f.write(f"  Total Ground Truth: {m['total_ground_truths']}\n")
                    f.write(f"  Total Predictions: {m['total_predictions']}\n")
                
                f.write("\n")
            
            f.write("=" * 70 + "\n")

def main():
    """Main execution function"""
    # Configuration
    model_path = '/content/Task1_VehicleDetection/weights/parking_detector_20250610_075747/weights/best.pt'
    test_data_path = '/content/Task1_VehicleDetection/data/test'  # Adjust this path
    
    # Create evaluator
    evaluator = ParkingDetectorEvaluator(model_path, test_data_path)
    
    # Run evaluation
    print("Starting model evaluation...")
    results = evaluator.evaluate_model(conf_threshold=0.5, iou_threshold=0.5)
    
    if results:
        # Print results
        evaluator.print_results()
        
        # Create visualizations
        evaluator.visualize_results()
        
        # Save results
        evaluator.save_results()
        
        print("\nEvaluation completed successfully!")
    else:
        print("Evaluation failed. Please check your test data path and format.")

if __name__ == "__main__":
    main()
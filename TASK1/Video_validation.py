# save as validate_complete.py
import argparse
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def non_max_suppression_custom(detections, iou_thresh=0.45):
    """Custom NMS to handle multi-scale detections"""
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        
        # Remove overlapping detections
        detections = [d for d in detections if calculate_iou(best['box'], d['box']) < iou_thresh]
    
    return keep

class EdgeAwareParkingDetector:
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
    def preprocess_frame(self, frame):
        """Advanced preprocessing for edge detection"""
        # Store original
        original = frame.copy()
        
        # 1. Adaptive histogram equalization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        frame = cv2.merge([l, a, b])
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        
        # 2. Edge-aware smoothing
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # 3. Enhance contrast at edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        frame = cv2.addWeighted(frame, 0.8, edges_3channel, 0.2, 0)
        
        return frame
    
    def sliding_window_detection(self, frame, window_size=(640, 640), overlap=0.3):
        """Sliding window approach for comprehensive detection"""
        h, w = frame.shape[:2]
        detections = []
        
        step_x = int(window_size[0] * (1 - overlap))
        step_y = int(window_size[1] * (1 - overlap))
        
        # Ensure we cover the entire image including edges
        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                # Calculate window boundaries
                x_end = min(x + window_size[0], w)
                y_end = min(y + window_size[1], h)
                
                # Skip if window is too small
                if (x_end - x) < window_size[0] * 0.5 or (y_end - y) < window_size[1] * 0.5:
                    continue
                
                # Extract window
                window = frame[y:y_end, x:x_end]
                
                # Pad window if necessary
                if window.shape[0] < window_size[1] or window.shape[1] < window_size[0]:
                    window = cv2.copyMakeBorder(
                        window,
                        0, window_size[1] - window.shape[0],
                        0, window_size[0] - window.shape[1],
                        cv2.BORDER_REPLICATE
                    )
                
                # Run detection
                results = self.model(window, conf=self.conf_thresh, verbose=False)
                
                # Adjust coordinates to full frame
                for r in results:
                    if r.boxes is not None:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        boxes[:, [0, 2]] += x  # Adjust x coordinates
                        boxes[:, [1, 3]] += y  # Adjust y coordinates
                        
                        # Clip to image boundaries
                        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
                        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
                        
                        for i, box in enumerate(boxes):
                            detections.append({
                                'box': box,
                                'conf': r.boxes.conf[i].cpu().numpy(),
                                'class': int(r.boxes.cls[i].cpu().numpy())
                            })
        
        return detections
    
    def detect_with_padding(self, frame, pad_size=64):
        """Detect with padding to catch edge cases"""
        # Add padding
        padded = cv2.copyMakeBorder(
            frame, pad_size, pad_size, pad_size, pad_size,
            cv2.BORDER_REPLICATE
        )
        
        # Detect
        results = self.model(padded, conf=self.conf_thresh, verbose=False)
        detections = []
        
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                # Adjust for padding
                boxes[:, [0, 2]] -= pad_size
                boxes[:, [1, 3]] -= pad_size
                
                # Clip to original image boundaries
                h, w = frame.shape[:2]
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
                
                for i, box in enumerate(boxes):
                    # Skip invalid boxes
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                        
                    detections.append({
                        'box': box,
                        'conf': r.boxes.conf[i].cpu().numpy(),
                        'class': int(r.boxes.cls[i].cpu().numpy())
                    })
        
        return detections

def parse_args():
    parser = argparse.ArgumentParser(description='Complete Parking Detection with Edge Enhancement')
    parser.add_argument('--weight', type=str, required=True, help='Path to model')
    parser.add_argument('--video', type=str, required=True, help='Path to video')
    parser.add_argument('--output', type=str, default='output_complete.mp4')
    parser.add_argument('--conf', type=float, default=0.2, help='Confidence threshold')
    parser.add_argument('--method', type=str, default='all', 
                       choices=['standard', 'sliding', 'padding', 'all'],
                       help='Detection method')
    parser.add_argument('--show-process', action='store_true', help='Show preprocessing')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check files
    if not os.path.exists(args.weight):
        print(f"Error: Model file not found: {args.weight}")
        return
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Initialize detector
    print(f"Loading model: {args.weight}")
    detector = EdgeAwareParkingDetector(args.weight, args.conf)
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = []
    
    print(f"Processing video with method: {args.method}")
    print(f"Total frames: {total_frames}")
    print(f"Confidence threshold: {args.conf}")
    print("-" * 50)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        original_frame = frame.copy()
        
        # Preprocess
        processed = detector.preprocess_frame(frame)
        
        # Detect based on method
        all_detections = []
        
        if args.method in ['standard', 'all']:
            # Standard detection with preprocessing
            results = detector.model(processed, conf=args.conf, verbose=False)
            for r in results:
                if r.boxes is not None:
                    for i in range(len(r.boxes)):
                        all_detections.append({
                            'box': r.boxes.xyxy[i].cpu().numpy(),
                            'conf': r.boxes.conf[i].cpu().numpy(),
                            'class': int(r.boxes.cls[i].cpu().numpy())
                        })
        
        if args.method in ['padding', 'all']:
            # Detection with padding
            detections = detector.detect_with_padding(processed)
            all_detections.extend(detections)
        
        if args.method in ['sliding', 'all']:
            # Sliding window detection
            detections = detector.sliding_window_detection(processed)
            all_detections.extend(detections)
        
        # Apply NMS to remove duplicates
        if all_detections:
            all_detections = non_max_suppression_custom(all_detections, 0.3)
        
        # Draw detections on original frame
        occupied = 0
        empty = 0
        
        for det in all_detections:
            x1, y1, x2, y2 = det['box']
            cls = det['class']
            conf = det['conf']
            
            # Get class name
            class_name = detector.model.names[cls]
            
            if 'occupied' in class_name.lower():
                occupied += 1
                color = (0, 0, 255)  # Red
            else:
                empty += 1
                color = (0, 255, 0)  # Green
            
            # Draw rectangle
            cv2.rectangle(original_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(original_frame, 
                         (int(x1), int(y1) - 20), 
                         (int(x1) + label_size[0], int(y1)), 
                         color, -1)
            cv2.putText(original_frame, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add summary
        total = occupied + empty
        total_detections.append(total)
        
        # Info panel
        info_panel = np.zeros((80, width, 3), dtype=np.uint8)
        cv2.putText(info_panel, f"Frame: {frame_count}/{total_frames} | Method: {args.method}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_panel, f"Occupied: {occupied} | Empty: {empty} | Total: {total}", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine frame with info panel
        if args.show_process:
            # Show original and processed side by side
            combined = np.hstack([original_frame, processed])
            combined = cv2.resize(combined, (width, height))
        else:
            combined = original_frame
        
        # Write frame
        out.write(combined)
        
        # Progress
        if frame_count % 30 == 0:
            avg_detections = np.mean(total_detections[-30:]) if len(total_detections) >= 30 else np.mean(total_detections)
            print(f"Progress: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) - "
                  f"Current: {total} spaces (Occupied: {occupied}, Empty: {empty}) - "
                  f"Avg: {avg_detections:.1f} spaces")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Final statistics
    if total_detections:
        avg_total = np.mean(total_detections)
        max_total = max(total_detections)
        min_total = min(total_detections)
        
        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Average parking spaces detected per frame: {avg_total:.2f}")
        print(f"Maximum spaces detected in a frame: {max_total}")
        print(f"Minimum spaces detected in a frame: {min_total}")
        print(f"Detection method used: {args.method}")
        print(f"Output saved to: {args.output}")
        print("="*60)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()

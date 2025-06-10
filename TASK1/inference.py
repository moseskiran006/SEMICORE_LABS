import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import json
from datetime import datetime
import time

class ParkingSpaceDetector:
    def __init__(self, model_path='/content/Task1_VehicleDetection/weights/parking_detector_20250610_075747/weights/best.pt', conf_threshold=0.5):
        """Initialize the parking space detector"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Class names
        self.class_names = {0: 'occupied', 1: 'empty'}
        
        # Colors for visualization
        self.colors = {
            'occupied': (0, 0, 255),    # Red
            'empty': (0, 255, 0)        # Green
        }
        
    def detect_parking_spaces(self, image_path):
        """Detect parking spaces in an image"""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Run inference
        start_time = time.time()
        results = self.model(image, conf=self.conf_threshold, device=self.device)
        inference_time = time.time() - start_time
        
        # Process results
        detections = results[0]
        
        # Count occupied and empty spaces
        occupied_count = 0
        empty_count = 0
        
        # Draw bounding boxes
        annotated_image = image.copy()
        
        if len(detections.boxes) > 0:
            boxes = detections.boxes.xyxy.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy()
            confidences = detections.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                class_name = self.class_names[int(cls)]
                
                if class_name == 'occupied':
                    occupied_count += 1
                # Task1_VehicleDetection/inference.py (continued)

                else:
                    empty_count += 1
                
                # Draw bounding box
                color = self.colors[class_name]
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 4), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add summary text
        summary_text = f"Occupied: {occupied_count} | Empty: {empty_count} | Total: {occupied_count + empty_count}"
        cv2.putText(annotated_image, summary_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Calculate FPS
        fps = 1.0 / inference_time
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(annotated_image, fps_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        results_dict = {
            'occupied_spaces': occupied_count,
            'empty_spaces': empty_count,
            'total_spaces': occupied_count + empty_count,
            'inference_time': inference_time,
            'fps': fps,
            'timestamp': datetime.now().isoformat()
        }
        
        return annotated_image, results_dict
    
    def process_video(self, video_path, output_path=None, display=False):
        """Process a video file and detect parking spaces frame by frame"""
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        results_log = []
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            start_time = time.time()
            results = self.model(frame, conf=self.conf_threshold, device=self.device)
            inference_time = time.time() - start_time
            
            # Process results
            detections = results[0]
            occupied_count = 0
            empty_count = 0
            
            # Annotate frame
            annotated_frame = frame.copy()
            
            if len(detections.boxes) > 0:
                boxes = detections.boxes.xyxy.cpu().numpy()
                classes = detections.boxes.cls.cpu().numpy()
                confidences = detections.boxes.conf.cpu().numpy()
                
                for box, cls, conf in zip(boxes, classes, confidences):
                    x1, y1, x2, y2 = box.astype(int)
                    class_name = self.class_names[int(cls)]
                    
                    if class_name == 'occupied':
                        occupied_count += 1
                    else:
                        empty_count += 1
                    
                    # Draw bounding box
                    color = self.colors[class_name]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add summary information
            info_text = [
                f"Frame: {frame_count}/{total_frames}",
                f"Occupied: {occupied_count}",
                f"Empty: {empty_count}",
                f"Total: {occupied_count + empty_count}",
                f"FPS: {1/inference_time:.2f}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(annotated_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 25
            
            # Log results
            frame_result = {
                'frame': frame_count,
                'occupied': occupied_count,
                'empty': empty_count,
                'total': occupied_count + empty_count,
                'inference_time': inference_time
            }
            results_log.append(frame_result)
            
            # Write frame if output specified
            if output_path:
                out.write(annotated_frame)
            
            # Display if requested
            if display:
                cv2.imshow('Parking Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        if display:
            cv2.destroyAllWindows()
        
        # Calculate average statistics
        avg_occupied = sum(r['occupied'] for r in results_log) / len(results_log)
        avg_empty = sum(r['empty'] for r in results_log) / len(results_log)
        avg_inference_time = sum(r['inference_time'] for r in results_log) / len(results_log)
        
        summary = {
            'video_path': str(video_path),
            'total_frames': total_frames,
            'average_occupied': avg_occupied,
            'average_empty': avg_empty,
            'average_total_spaces': avg_occupied + avg_empty,
            'average_fps': 1 / avg_inference_time,
            'frame_results': results_log
        }
        
        return summary
    
    def benchmark_performance(self, test_images_dir):
        """Benchmark model performance on CPU and GPU"""
        test_images = list(Path(test_images_dir).glob('*.jpg')) + \
                     list(Path(test_images_dir).glob('*.png'))
        
        if not test_images:
            print("No test images found!")
            return
        
        print(f"Benchmarking on {len(test_images)} images...")
        
        # Test on current device
        inference_times = []
        
        for img_path in test_images[:10]:  # Test on first 10 images
            img = cv2.imread(str(img_path))
            
            # Warm-up
            _ = self.model(img, device=self.device)
            
            # Actual timing
            start = time.time()
            _ = self.model(img, device=self.device)
            inference_time = time.time() - start
            inference_times.append(inference_time)
        
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        fps = 1.0 / avg_time
        
        # Get model size
        model_size = Path(self.model.model.pt_path).stat().st_size / (1024 * 1024)  # MB
        
        print(f"\nPerformance Metrics:")
        print(f"Device: {self.device}")
        print(f"Average inference time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"Average FPS: {fps:.2f}")
        print(f"Model size: {model_size:.2f} MB")
        
        return {
            'device': self.device,
            'avg_inference_time_ms': avg_time * 1000,
            'std_inference_time_ms': std_time * 1000,
            'fps': fps,
            'model_size_mb': model_size
        }

# Main execution
if __name__ == "__main__":
    # Initialize detector
    detector = ParkingSpaceDetector(
        model_path='/content/Task1_VehicleDetection/weights/parking_detector_20250610_075747/weights/best.pt',
        conf_threshold=0.5
    )
    
    # Test on single image
    test_image = '/content/Task1_VehicleDetection/detect.jpg'
    if Path(test_image).exists():
        annotated_img, results = detector.detect_parking_spaces(test_image)
        
        print("\nDetection Results:")
        print(f"Occupied spaces: {results['occupied_spaces']}")
        print(f"Empty spaces: {results['empty_spaces']}")
        print(f"Total spaces: {results['total_spaces']}")
        print(f"FPS: {results['fps']:.2f}")
        
        # Save annotated image
        cv2.imwrite('./results/detected_parking.jpg', annotated_img)
    
    # Benchmark performance
    print("\nBenchmarking model performance...")
    perf_metrics = detector.benchmark_performance('./data/test/images')
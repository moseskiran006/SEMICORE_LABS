import torch
from ultralytics import YOLO
import cv2
import psutil
import time
import numpy as np
from pathlib import Path

model = YOLO("models/best3.pt")

def run_inference(image_path):
    start = time.time()
    results = model(image_path)
    end = time.time()

    fps = round(1 / (end - start), 2)
    
    # Get original image
    img = cv2.imread(image_path)
    
    # Analyze parking spaces
    parking_stats = count_vehicles(results[0])
    
    # Plot results with parking info
    result_img = results[0].plot()
    result_img = add_parking_info(result_img, parking_stats)
    
    output_path = 'static/result.jpg'
    cv2.imwrite(output_path, result_img)

    return output_path, fps, parking_stats

def run_video_inference(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    output_path = 'static/result_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_time = 0
    total_empty = 0
    total_occupied = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # Run inference
        results = model(frame)
        parking_stats = count_vehicles(results[0])
        
        # Plot results
        result_frame = results[0].plot()
        result_frame = add_parking_info(result_frame, parking_stats)
        
        out.write(result_frame)
        
        end_time = time.time()
        total_time += (end_time - start_time)
        total_empty += parking_stats["empty"]
        total_occupied += parking_stats["occupied"]
        frame_count += 1
    
    cap.release()
    out.release()
    
    avg_fps = round(frame_count / total_time, 2) if total_time > 0 else 0
    avg_parking_stats = {
        "empty": round(total_empty / frame_count),
        "occupied": round(total_occupied / frame_count),
        "total": round((total_empty + total_occupied) / frame_count)
    }
    
    return output_path, avg_fps, avg_parking_stats

def analyze_parking_spaces(frame):
    """Analyze parking spaces in real-time for webcam"""
    results = model(frame)
    parking_stats = count_vehicles(results[0])
    
    # Plot results
    result_frame = results[0].plot()
    result_frame = add_parking_info(result_frame, parking_stats)
    
    return result_frame, parking_stats

def count_vehicles(results):
    """Count vehicles and estimate parking spaces"""
    # Get detected vehicles
    vehicles = 0
    if results.boxes is not None:
        vehicles = len(results.boxes)
    
    # Estimate parking spaces (you can customize this logic)
    # For now, assuming a simple estimation based on image area and vehicle count
    estimated_total_spaces = max(20, vehicles + 5)  # Minimum 20 spaces
    occupied = vehicles
    empty = estimated_total_spaces - occupied
    
    return {
        "occupied": occupied,
        "empty": empty,
        "total": estimated_total_spaces
    }

def add_parking_info(img, parking_stats):
    """Add parking information overlay to image"""
    height, width = img.shape[:2]
    
    # Create info panel
    panel_height = 120
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    panel.fill(50)  # Dark gray background
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Parking statistics
    occupied_text = f"Occupied: {parking_stats['occupied']}"
    empty_text = f"Empty: {parking_stats['empty']}"
    total_text = f"Total: {parking_stats['total']}"
    
    # Calculate text positions
    y_offset = 30
    cv2.putText(panel, occupied_text, (20, y_offset), font, font_scale, (0, 0, 255), thickness)  # Red
    cv2.putText(panel, empty_text, (200, y_offset), font, font_scale, (0, 255, 0), thickness)    # Green
    cv2.putText(panel, total_text, (350, y_offset), font, font_scale, (255, 255, 255), thickness) # White
    
    # Add occupancy percentage
    occupancy_rate = (parking_stats['occupied'] / parking_stats['total']) * 100
    occupancy_text = f"Occupancy: {occupancy_rate:.1f}%"
    cv2.putText(panel, occupancy_text, (20, y_offset + 40), font, font_scale, (255, 255, 0), thickness)  # Yellow
    
    # Status indicator
    if occupancy_rate > 90:
        status = "FULL"
        status_color = (0, 0, 255)  # Red
    elif occupancy_rate > 70:
        status = "BUSY"
        status_color = (0, 165, 255)  # Orange
    else:
        status = "AVAILABLE"
        status_color = (0, 255, 0)  # Green
    
    cv2.putText(panel, f"Status: {status}", (200, y_offset + 40), font, font_scale, status_color, thickness)
    
    # Combine image with info panel
    result_img = np.vstack([img, panel])
    
    return result_img

def get_system_metrics():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpu = round(gpus[0].load * 100, 2) if gpus else "N/A"
    except:
        gpu = "N/A"
    return cpu, mem, gpu
import argparse
import os
import cv2
from ultralytics import YOLO

def validate_model(weights_path, test_video_path, output_dir, aspect_ratio_threshold=1.0):
    
    # Validate input path
    if not os.path.exists(test_video_path):
        raise FileNotFoundError(f"Video file does not exist: {test_video_path}")
    if not test_video_path.lower().endswith(('.avi', '.mp4')):
        raise ValueError(f"Invalid video file: {test_video_path}. Must be .avi or .mp4")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    try:
        model = YOLO(weights_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {weights_path}: {e}")

    # Open input video
    cap = cv2.VideoCapture(test_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {test_video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up output video
    video_base = os.path.splitext(os.path.basename(test_video_path))[0]
    output_video_path = os.path.join(output_dir, f"{video_base}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing video: {test_video_path}")
    fallen_detected = False
    video_status = "Not Fallen"  # Tracks persistent status
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)

        # Process detections
        frame_status = "Not Fallen"
        for det in results[0].boxes:
            # Get bounding box dimensions
            x, y, w, h = det.xywh[0].numpy()
            # Classify as Fallen if horizontal
            if w / h > aspect_ratio_threshold:
                frame_status = "Fallen"
                fallen_detected = True
                video_status = "Fallen"  # Lock status to Fallen
            else:
                frame_status = "Not Fallen"

            # Draw bounding box and label
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            color = (0, 0, 255) if frame_status == "Fallen" else (0, 255, 0)  # Red for Fallen, green for Not Fallen
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, frame_status, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Use persistent video_status for status bar
        bar_color = (0, 0, 255) if video_status == "Fallen" else (0, 255, 0)
        bar_height = 40
        cv2.rectangle(frame, (0, height - bar_height), (width, height), bar_color, -1)  # Filled rectangle
        cv2.putText(frame, video_status, (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White text

        # Write frame to output video
        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    # Release resources
    cap.release()
    out.release()

    # Print final result
    final_result = "Fallen detected" if fallen_detected else "No fallen detected"
    print(f"\nValidation completed. Output video saved to: {output_video_path}")
    print(f"Final result: {final_result}")

    return final_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLOv11n model to detect fallen status in a video")
    parser.add_argument("--weights", required=True, help="Path to trained model weights")
    parser.add_argument("--test_video", required=True, help="Path to test video")
    parser.add_argument("--output", default="validation_results", help="Output directory")
    parser.add_argument("--aspect_ratio", type=float, default=1.0, help="Width/height ratio threshold")
    args = parser.parse_args()

    try:
        validate_model(args.weights, args.test_video, args.output, args.aspect_ratio)
    except Exception as e:
        print(f"Validation failed: {e}")

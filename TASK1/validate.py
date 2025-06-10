#!/usr/bin/env python3
"""
CLI script for validating parking detection model on images and videos

Usage:
    python validate_model.py --weights path/to/weights.pt --test_videos folder
    python validate_model.py --weights path/to/weights.pt --test_images folder
    python validate_model.py --weights path/to/weights.pt --test_image single_image.jpg
    python validate_model.py --weights path/to/weights.pt --test_video single_video.mp4
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import ParkingSpaceDetector

class ParkingModelValidator:
    """CLI validator for parking detection model"""
    
    def __init__(self, weights_path, conf_threshold=0.5, save_outputs=True):
        """Initialize validator with model weights"""
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.save_outputs = save_outputs
        
        # Create output directory
        self.output_dir = Path(f'validation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize detector
        print(f"Loading model from: {weights_path}")
        self.detector = ParkingSpaceDetector(
            model_path=weights_path,
            conf_threshold=conf_threshold
        )
        print(f"Model loaded successfully. Using device: {self.detector.device}")
        
        # Results storage
        self.results = {
            'images': [],
            'videos': [],
            'summary': {}
        }
    
    def validate_single_image(self, image_path, save_output=True):
        """Validate model on a single image"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return None
        
        print(f"\nProcessing image: {image_path.name}")
        
        # Perform detection
        annotated_image, results = self.detector.detect_parking_spaces(str(image_path))
        
        # Add image name to results
        results['image_name'] = image_path.name
        results['image_path'] = str(image_path)
        
        # Save annotated image if requested
        if save_output and self.save_outputs:
            output_path = self.output_dir / 'images' / f'detected_{image_path.name}'
            output_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(output_path), annotated_image)
            results['output_path'] = str(output_path)
        
        # Print results
        print(f"  Occupied spaces: {results['occupied_spaces']}")
        print(f"  Empty spaces: {results['empty_spaces']}")
        print(f"  Total spaces: {results['total_spaces']}")
        print(f"  Inference time: {results['inference_time']*1000:.2f} ms")
        print(f"  FPS: {results['fps']:.2f}")
        
        return results
    
    def validate_image_folder(self, folder_path):
        """Validate model on all images in a folder"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"Error: Folder not found: {folder_path}")
            return []
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f'*{ext}'))
            image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {folder_path}")
            return []
        
        print(f"\nFound {len(image_files)} images in {folder_path}")
        
        # Process each image
        results = []
        for img_path in tqdm(image_files, desc="Processing images"):
            result = self.validate_single_image(img_path, save_output=True)
            if result:
                results.append(result)
                self.results['images'].append(result)
        
        # Generate summary
        self._generate_image_summary(results)
        
        return results
    
    def validate_single_video(self, video_path, save_output=True, sample_rate=1):
        """Validate model on a single video"""
        video_path = Path(video_path)
        
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}")
            return None
        
        print(f"\nProcessing video: {video_path.name}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if saving output
        out_writer = None
        if save_output and self.save_outputs:
            output_path = self.output_dir / 'videos' / f'detected_{video_path.name}'
            output_path.parent.mkdir(exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process video
        frame_results = []
        frame_count = 0
        
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame based on sample_rate
            if frame_count % sample_rate == 0:
                # Run detection
                start_time = time.time()
                results = self.detector.model(frame, conf=self.detector.conf_threshold)
                inference_time = time.time() - start_time
                
                # Count detections
                occupied_count = 0
                empty_count = 0
                
                # Annotate frame
                annotated_frame = frame.copy()
                
                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, cls, conf in zip(boxes, classes, confidences):
                        x1, y1, x2, y2 = box.astype(int)
                        class_name = self.detector.class_names[int(cls)]
                        
                        if class_name == 'occupied':
                            occupied_count += 1
                            color = (0, 0, 255)
                        else:
                            empty_count += 1
                            color = (0, 255, 0)
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated_frame, f"{class_name}: {conf:.2f}", 
                                   (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add frame info
                info_text = f"Frame: {frame_count} | Occupied: {occupied_count} | Empty: {empty_count}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Save frame results
                frame_result = {
                    'frame': frame_count,
                    'occupied': occupied_count,
                    'empty': empty_count,
                    'total': occupied_count + empty_count,
                    'inference_time': inference_time,
                    'fps': 1.0 / inference_time
                }
                frame_results.append(frame_result)
                
                # Write frame if saving
                if out_writer:
                    out_writer.write(annotated_frame)
            else:
                # For non-sampled frames, just write the original
                if out_writer:
                    out_writer.write(frame)
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if out_writer:
            out_writer.release()
        
        # Calculate summary statistics
        if frame_results:
            avg_occupied = np.mean([r['occupied'] for r in frame_results])
            avg_empty = np.mean([r['empty'] for r in frame_results])
            avg_total = np.mean([r['total'] for r in frame_results])
            avg_inference = np.mean([r['inference_time'] for r in frame_results])
            avg_fps = np.mean([r['fps'] for r in frame_results])
            
            video_summary = {
                'video_name': video_path.name,
                'video_path': str(video_path),
                'total_frames': total_frames,
                'processed_frames': len(frame_results),
                'avg_occupied': avg_occupied,
                'avg_empty': avg_empty,
                'avg_total': avg_total,
                'avg_inference_time': avg_inference,
                'avg_fps': avg_fps,
                'frame_results': frame_results
            }
            
            if save_output and self.save_outputs:
                video_summary['output_path'] = str(output_path)
            
            print(f"\n  Video Summary:")
            print(f"    Average occupied spaces: {avg_occupied:.1f}")
            print(f"    Average empty spaces: {avg_empty:.1f}")
            print(f"    Average total spaces: {avg_total:.1f}")
            print(f"    Average inference time: {avg_inference*1000:.2f} ms")
            print(f"    Average FPS: {avg_fps:.2f}")
            
            return video_summary
        
        return None
    
    def validate_video_folder(self, folder_path, sample_rate=1):
        """Validate model on all videos in a folder"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"Error: Folder not found: {folder_path}")
            return []
        
        # Get all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(folder_path.glob(f'*{ext}'))
            video_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"No videos found in {folder_path}")
            return []
        
        print(f"\nFound {len(video_files)} videos in {folder_path}")
        
        # Process each video
        results = []
        for video_path in video_files:
            result = self.validate_single_video(video_path, save_output=True, sample_rate=sample_rate)
            if result:
                results.append(result)
                self.results['videos'].append(result)
        
        # Generate summary
        self._generate_video_summary(results)
        
        return results
    
    def _generate_image_summary(self, results):
        """Generate summary statistics for image results"""
        # validate_model.py (continued)

        if not results:
            return
        
        total_images = len(results)
        total_occupied = sum(r['occupied_spaces'] for r in results)
        total_empty = sum(r['empty_spaces'] for r in results)
        total_spaces = sum(r['total_spaces'] for r in results)
        avg_inference = np.mean([r['inference_time'] for r in results])
        avg_fps = np.mean([r['fps'] for r in results])
        
        summary = {
            'total_images': total_images,
            'total_occupied_spaces': total_occupied,
            'total_empty_spaces': total_empty,
            'total_parking_spaces': total_spaces,
            'avg_spaces_per_image': total_spaces / total_images if total_images > 0 else 0,
            'avg_inference_time_ms': avg_inference * 1000,
            'avg_fps': avg_fps
        }
        
        self.results['summary']['images'] = summary
        
        print(f"\n{'='*50}")
        print("IMAGE VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total images processed: {total_images}")
        print(f"Total occupied spaces detected: {total_occupied}")
        print(f"Total empty spaces detected: {total_empty}")
        print(f"Total parking spaces: {total_spaces}")
        print(f"Average spaces per image: {summary['avg_spaces_per_image']:.1f}")
        print(f"Average inference time: {summary['avg_inference_time_ms']:.2f} ms")
        print(f"Average FPS: {summary['avg_fps']:.2f}")
        print(f"{'='*50}\n")
    
    def _generate_video_summary(self, results):
        """Generate summary statistics for video results"""
        if not results:
            return
        
        total_videos = len(results)
        total_frames = sum(r['total_frames'] for r in results)
        processed_frames = sum(r['processed_frames'] for r in results)
        avg_occupied = np.mean([r['avg_occupied'] for r in results])
        avg_empty = np.mean([r['avg_empty'] for r in results])
        avg_total = np.mean([r['avg_total'] for r in results])
        avg_fps = np.mean([r['avg_fps'] for r in results])
        
        summary = {
            'total_videos': total_videos,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'avg_occupied_spaces': avg_occupied,
            'avg_empty_spaces': avg_empty,
            'avg_total_spaces': avg_total,
            'avg_fps': avg_fps
        }
        
        self.results['summary']['videos'] = summary
        
        print(f"\n{'='*50}")
        print("VIDEO VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total videos processed: {total_videos}")
        print(f"Total frames: {total_frames}")
        print(f"Processed frames: {processed_frames}")
        print(f"Average occupied spaces: {avg_occupied:.1f}")
        print(f"Average empty spaces: {avg_empty:.1f}")
        print(f"Average total spaces: {avg_total:.1f}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"{'='*50}\n")
    
    def save_results(self):
        """Save all results to files"""
        # Save JSON results
        json_path = self.output_dir / 'validation_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {json_path}")
        
        # Save CSV for images
        if self.results['images']:
            df_images = pd.DataFrame(self.results['images'])
            csv_path = self.output_dir / 'image_results.csv'
            df_images.to_csv(csv_path, index=False)
            print(f"Image results CSV saved to: {csv_path}")
        
        # Save CSV for videos
        if self.results['videos']:
            # Create summary CSV
            video_summaries = []
            for video in self.results['videos']:
                summary = {k: v for k, v in video.items() if k != 'frame_results'}
                video_summaries.append(summary)
            
            df_videos = pd.DataFrame(video_summaries)
            csv_path = self.output_dir / 'video_results.csv'
            df_videos.to_csv(csv_path, index=False)
            print(f"Video results CSV saved to: {csv_path}")
            
            # Save detailed frame results for each video
            for video in self.results['videos']:
                if 'frame_results' in video:
                    video_name = Path(video['video_name']).stem
                    df_frames = pd.DataFrame(video['frame_results'])
                    frame_csv = self.output_dir / f'video_frames_{video_name}.csv'
                    df_frames.to_csv(frame_csv, index=False)
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Generate HTML report
        self._generate_html_report()
    
    def _generate_visualizations(self):
        """Generate visualization plots"""
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Image results visualization
        if self.results['images']:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            df = pd.DataFrame(self.results['images'])
            
            # Occupied vs Empty spaces distribution
            axes[0, 0].scatter(df['occupied_spaces'], df['empty_spaces'], alpha=0.6)
            axes[0, 0].set_xlabel('Occupied Spaces')
            axes[0, 0].set_ylabel('Empty Spaces')
            axes[0, 0].set_title('Occupied vs Empty Spaces Distribution')
            
            # Total spaces histogram
            axes[0, 1].hist(df['total_spaces'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_xlabel('Total Spaces')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Total Parking Spaces')
            
            # Inference time distribution
            axes[1, 0].hist(df['inference_time'] * 1000, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].set_xlabel('Inference Time (ms)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Inference Time Distribution')
            
            # FPS distribution
            axes[1, 1].hist(df['fps'], bins=20, alpha=0.7, color='salmon', edgecolor='black')
            axes[1, 1].set_xlabel('FPS')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('FPS Distribution')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'image_results_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Video results visualization
        if self.results['videos']:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Collect all frame results
            all_frames = []
            for video in self.results['videos']:
                if 'frame_results' in video:
                    for frame in video['frame_results']:
                        frame['video'] = video['video_name']
                        all_frames.append(frame)
            
            if all_frames:
                df_frames = pd.DataFrame(all_frames)
                
                # Parking spaces over time (for first video)
                first_video = self.results['videos'][0]['video_name']
                first_video_frames = df_frames[df_frames['video'] == first_video]
                
                axes[0, 0].plot(first_video_frames['frame'], first_video_frames['occupied'], 
                               label='Occupied', color='red', linewidth=2)
                axes[0, 0].plot(first_video_frames['frame'], first_video_frames['empty'], 
                               label='Empty', color='green', linewidth=2)
                axes[0, 0].set_xlabel('Frame Number')
                axes[0, 0].set_ylabel('Number of Spaces')
                axes[0, 0].set_title(f'Parking Spaces Over Time - {first_video}')
                axes[0, 0].legend()
                
                # Average spaces per video
                video_avg = df_frames.groupby('video')[['occupied', 'empty', 'total']].mean()
                x = np.arange(len(video_avg))
                width = 0.25
                
                axes[0, 1].bar(x - width, video_avg['occupied'], width, label='Occupied', color='red', alpha=0.7)
                axes[0, 1].bar(x, video_avg['empty'], width, label='Empty', color='green', alpha=0.7)
                axes[0, 1].bar(x + width, video_avg['total'], width, label='Total', color='blue', alpha=0.7)
                axes[0, 1].set_xlabel('Video')
                axes[0, 1].set_ylabel('Average Spaces')
                axes[0, 1].set_title('Average Parking Spaces per Video')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels([Path(v).stem for v in video_avg.index], rotation=45, ha='right')
                axes[0, 1].legend()
                
                # FPS over frames
                axes[1, 0].scatter(df_frames['frame'], df_frames['fps'], alpha=0.5, s=10)
                axes[1, 0].set_xlabel('Frame Number')
                axes[1, 0].set_ylabel('FPS')
                axes[1, 0].set_title('FPS Performance Over Frames')
                
                # Inference time distribution for videos
                axes[1, 1].hist(df_frames['inference_time'] * 1000, bins=30, alpha=0.7, 
                               color='orange', edgecolor='black')
                axes[1, 1].set_xlabel('Inference Time (ms)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Video Inference Time Distribution')
            
            plt.tight_layout()
            plt.savefig(viz_dir / 'video_results_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_html_report(self):
        """Generate an HTML report with results"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Parking Detection Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h3 {{ color: #888; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; text-align: center; }}
                .metric-value {{ font-size: 36px; font-weight: bold; color: #2196F3; }}
                .metric-label {{ color: #666; font-size: 14px; margin-top: 5px; }}
                .info {{ background-color: #e3f2fd; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
                .visualization img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .footer {{ text-align: center; color: #999; margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚗 Parking Detection Validation Report</h1>
                
                <div class="info">
                    <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Model weights:</strong> {self.weights_path}</p>
                    <p><strong>Confidence threshold:</strong> {self.conf_threshold}</p>
                    <p><strong>Device:</strong> {self.detector.device}</p>
                </div>
        """
        
        # Add image summary if available
        if 'images' in self.results['summary']:
            summary = self.results['summary']['images']
            html_content += f"""
                <h2>📸 Image Validation Results</h2>
                <div class="summary">
                    <div class="metric">
                        # validate_model.py (continued)

                        <div class="metric-value">{summary['total_images']}</div>
                        <div class="metric-label">Total Images</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{summary['total_occupied_spaces']}</div>
                        <div class="metric-label">Total Occupied Spaces</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{summary['total_empty_spaces']}</div>
                        <div class="metric-label">Total Empty Spaces</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{summary['avg_spaces_per_image']:.1f}</div>
                        <div class="metric-label">Avg Spaces/Image</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{summary['avg_fps']:.1f}</div>
                        <div class="metric-label">Average FPS</div>
                    </div>
                </div>
                
                <h3>Detailed Image Results</h3>
                <table>
                    <tr>
                        <th>Image</th>
                        <th>Occupied</th>
                        <th>Empty</th>
                        <th>Total</th>
                        <th>Inference Time (ms)</th>
                        <th>FPS</th>
                    </tr>
            """
            
            # Add each image result
            for img in self.results['images'][:50]:  # Limit to first 50 for readability
                html_content += f"""
                    <tr>
                        <td>{img['image_name']}</td>
                        <td>{img['occupied_spaces']}</td>
                        <td>{img['empty_spaces']}</td>
                        <td>{img['total_spaces']}</td>
                        <td>{img['inference_time']*1000:.2f}</td>
                        <td>{img['fps']:.2f}</td>
                    </tr>
                """
            
            if len(self.results['images']) > 50:
                html_content += f"""
                    <tr>
                        <td colspan="6" style="text-align: center; font-style: italic;">
                            ... and {len(self.results['images']) - 50} more images
                        </td>
                    </tr>
                """
            
            html_content += "</table>"
        
        # Add video summary if available
        if 'videos' in self.results['summary']:
            summary = self.results['summary']['videos']
            html_content += f"""
                <h2>🎥 Video Validation Results</h2>
                <div class="summary">
                    <div class="metric">
                        <div class="metric-value">{summary['total_videos']}</div>
                        <div class="metric-label">Total Videos</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{summary['avg_occupied_spaces']:.1f}</div>
                        <div class="metric-label">Avg Occupied Spaces</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{summary['avg_empty_spaces']:.1f}</div>
                        <div class="metric-label">Avg Empty Spaces</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{summary['avg_total_spaces']:.1f}</div>
                        <div class="metric-label">Avg Total Spaces</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{summary['avg_fps']:.1f}</div>
                        <div class="metric-label">Average FPS</div>
                    </div>
                </div>
                
                <h3>Detailed Video Results</h3>
                <table>
                    <tr>
                        <th>Video</th>
                        <th>Total Frames</th>
                        <th>Processed Frames</th>
                        <th>Avg Occupied</th>
                        <th>Avg Empty</th>
                        <th>Avg Total</th>
                        <th>Avg FPS</th>
                    </tr>
            """
            
            # Add each video result
            for video in self.results['videos']:
                html_content += f"""
                    <tr>
                        <td>{video['video_name']}</td>
                        <td>{video['total_frames']}</td>
                        <td>{video['processed_frames']}</td>
                        <td>{video['avg_occupied']:.1f}</td>
                        <td>{video['avg_empty']:.1f}</td>
                        <td>{video['avg_total']:.1f}</td>
                        <td>{video['avg_fps']:.1f}</td>
                    </tr>
                """
            
            html_content += "</table>"
        
        # Add visualizations if they exist
        viz_dir = self.output_dir / 'visualizations'
        if (viz_dir / 'image_results_analysis.png').exists():
            html_content += """
                <h2>📊 Visualizations</h2>
                <div class="visualization">
                    <h3>Image Results Analysis</h3>
                    <img src="visualizations/image_results_analysis.png" alt="Image Results Analysis">
                </div>
            """
        
        if (viz_dir / 'video_results_analysis.png').exists():
            html_content += """
                <div class="visualization">
                    <h3>Video Results Analysis</h3>
                    <img src="visualizations/video_results_analysis.png" alt="Video Results Analysis">
                </div>
            """
        
        # Add footer
        html_content += """
                <div class="footer">
                    <p>Generated by Parking Detection Validation Tool</p>
                    <p>Model: YOLOv11n | Framework: Ultralytics</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = self.output_dir / 'validation_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        print(f"HTML report saved to: {report_path}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='Validate parking detection model on images and videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on a single image
  python validate_model.py --weights ./weights/best.pt --test_image parking.jpg
  
  # Test on multiple images
  python validate_model.py --weights ./weights/best.pt --test_images ./test_images/
  
  # Test on a single video
  python validate_model.py --weights ./weights/best.pt --test_video parking_video.mp4
  
  # Test on multiple videos
  python validate_model.py --weights ./weights/best.pt --test_videos ./test_videos/
  
  # Test with custom confidence threshold
  python validate_model.py --weights ./weights/best.pt --test_images ./images/ --conf 0.6
  
  # Test videos with frame sampling (process every 5th frame)
  python validate_model.py --weights ./weights/best.pt --test_videos ./videos/ --sample_rate 5
  
  # Disable output saving
  python validate_model.py --weights ./weights/best.pt --test_images ./images/ --no_save
        """
    )
    
    # Required arguments
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights file')
    
    # Input options (at least one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--test_image', type=str,
                            help='Path to single test image')
    input_group.add_argument('--test_images', type=str,
                            help='Path to folder containing test images')
    input_group.add_argument('--test_video', type=str,
                            help='Path to single test video')
    input_group.add_argument('--test_videos', type=str,
                            help='Path to folder containing test videos')
    
    # Optional arguments
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='Sample rate for video processing (default: 1, process every frame)')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save annotated outputs')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                        help='Device to use for inference (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Validate weights path
    if not Path(args.weights).exists():
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)
    
    # Initialize validator
    print(f"\n{'='*60}")
    print("PARKING DETECTION MODEL VALIDATION")
    print(f"{'='*60}\n")
    
    validator = ParkingModelValidator(
        weights_path=args.weights,
        conf_threshold=args.conf,
        save_outputs=not args.no_save
    )
    
    # Process based on input type
    if args.test_image:
        # Single image
        result = validator.validate_single_image(args.test_image)
        if result:
            validator.results['images'].append(result)
            validator._generate_image_summary([result])
    
    elif args.test_images:
        # Multiple images
        validator.validate_image_folder(args.test_images)
    
    elif args.test_video:
        # Single video
        result = validator.validate_single_video(args.test_video, sample_rate=args.sample_rate)
        if result:
            validator.results['videos'].append(result)
            validator._generate_video_summary([result])
    
    elif args.test_videos:
        # Multiple videos
        validator.validate_video_folder(args.test_videos, sample_rate=args.sample_rate)
    
    # Save all results
    if validator.results['images'] or validator.results['videos']:
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")
        validator.save_results()
        
        print(f"\n{'='*60}")
        print("VALIDATION COMPLETE")
        print(f"{'='*60}")
        print(f"All results saved to: {validator.output_dir}")
        print(f"View the HTML report: {validator.output_dir}/validation_report.html")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
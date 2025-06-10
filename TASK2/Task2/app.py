from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import os
import cv2
from utils import run_inference, run_video_inference, get_system_metrics, analyze_parking_spaces
import threading
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for webcam streaming
camera = None
streaming = False
latest_frame = None
latest_stats = {"empty": 0, "occupied": 0, "fps": 0, "cpu": 0, "mem": 0, "gpu": "N/A"}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        detection_type = request.form.get('detection_type')
        
        if detection_type == 'image':
            image = request.files['image']
            if image:
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')
                image.save(image_path)
                
                output_path, fps, parking_stats = run_inference(image_path)
                cpu, mem, gpu = get_system_metrics()
                
                return render_template('index.html', 
                                     result_type='image',
                                     result_img=output_path, 
                                     fps=fps, 
                                     cpu=cpu, 
                                     mem=mem, 
                                     gpu=gpu,
                                     parking_stats=parking_stats)
        
        elif detection_type == 'video':
            video = request.files['video']
            if video:
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')
                video.save(video_path)
                
                output_path, avg_fps, parking_stats = run_video_inference(video_path)
                cpu, mem, gpu = get_system_metrics()
                
                return render_template('index.html', 
                                     result_type='video',
                                     result_video=output_path, 
                                     fps=avg_fps, 
                                     cpu=cpu, 
                                     mem=mem, 
                                     gpu=gpu,
                                     parking_stats=parking_stats)

    return render_template('index.html', result_type=None)

@app.route('/start_webcam')
def start_webcam():
    global camera, streaming
    if not streaming:
        camera = cv2.VideoCapture(0)
        streaming = True
        threading.Thread(target=webcam_stream, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/stop_webcam')
def stop_webcam():
    global camera, streaming
    streaming = False
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "stopped"})

@app.route('/webcam_feed')
def webcam_feed():
    def generate():
        global latest_frame
        while streaming and latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam_stats')
def webcam_stats():
    return jsonify(latest_stats)

def webcam_stream():
    global latest_frame, latest_stats, camera, streaming
    
    while streaming and camera is not None:
        ret, frame = camera.read()
        if ret:
            start_time = time.time()
            
            # Run inference on frame
            processed_frame, parking_stats = analyze_parking_spaces(frame)
            latest_frame = processed_frame
            
            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            
            # Get system metrics
            cpu, mem, gpu = get_system_metrics()
            
            # Update stats
            latest_stats.update({
                "empty": parking_stats["empty"],
                "occupied": parking_stats["occupied"],
                "fps": round(fps, 2),
                "cpu": cpu,
                "mem": mem,
                "gpu": gpu
            })
            
            time.sleep(0.1)  # Limit processing rate

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
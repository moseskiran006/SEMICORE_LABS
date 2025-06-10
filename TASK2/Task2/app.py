from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import os
import cv2
from utils import run_inference, run_video_inference, get_system_metrics, analyze_parking_spaces
import threading
import time
import socket
import base64
from PIL import Image
import io
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for webcam streaming
camera = None
streaming = False
latest_frame = None
latest_stats = {"empty": 0, "occupied": 0, "fps": 0, "cpu": 0, "mem": 0, "gpu": "N/A"}

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # Connect to a remote server to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

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

# Trick browsers into thinking it's secure
@app.route('/.well-known/acme-challenge/<path:path>')
def acme_challenge(path):
    """This tricks some browsers into thinking it's secure"""
    return "OK"

# Camera test page for debugging
@app.route('/camera-test')
def camera_test():
    """Simple camera test page"""
    local_ip = get_local_ip()
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Camera Permission Test</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 20px;
                max-width: 600px;
                margin: 0 auto;
            }}
            button {{
                background: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 10px 0;
            }}
            video {{
                width: 100%;
                max-width: 400px;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin: 10px 0;
            }}
            #status {{
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
            }}
            .success {{
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}
            .error {{
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }}
            .info {{
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }}
            .instructions {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <h1>Camera Permission Test</h1>
        
        <div class="instructions">
            <h3>📱 Mobile Users:</h3>
            <p>Current URL: <code>http://{local_ip}:5000</code></p>
            <p>If camera doesn't work, try:</p>
            <ol>
                <li><strong>Chrome Android:</strong> Go to <code>chrome://flags</code> → Search "Insecure origins" → Add <code>http://{local_ip}:5000</code></li>
                <li><strong>Use localhost:</strong> Enable USB debugging and use <code>adb reverse tcp:5000 tcp:5000</code></li>
                <li><strong>Use File Upload:</strong> Click "Take Photo" button below instead</li>
            </ol>
        </div>
        
        <button onclick="testCamera()">Test Live Camera Access</button>
        <button onclick="document.getElementById('fileInput').click()">📸 Take Photo (File Upload)</button>
        
        <input type="file" id="fileInput" accept="image/*" capture="environment" style="display:none;" onchange="handleFileUpload(event)">
        
        <div id="status"></div>
        <video id="video" autoplay playsinline style="display:none;"></video>
        <img id="capturedImage" style="width:100%; max-width:400px; display:none;">
        
        <script>
        async function testCamera() {{
            const status = document.getElementById('status');
            const video = document.getElementById('video');
            
            status.innerHTML = '<div class="info">Requesting camera access...</div>';
            
            try {{
                // Try different methods
                const methods = [
                    {{ video: true }},
                    {{ video: {{ facingMode: 'environment' }} }},
                    {{ video: {{ facingMode: 'user' }} }},
                    {{ video: {{ width: 1280, height: 720 }} }}
                ];
                
                let stream = null;
                let workingMethod = null;
                
                for (const constraints of methods) {{
                    try {{
                        stream = await navigator.mediaDevices.getUserMedia(constraints);
                        workingMethod = constraints;
                        break;
                    }} catch (e) {{
                        console.log('Method failed:', constraints, e);
                    }}
                }}
                
                if (stream) {{
                    video.srcObject = stream;
                    video.style.display = 'block';
                    status.innerHTML = '<div class="success">✓ Camera working! Method: ' + JSON.stringify(workingMethod) + '</div>';
                    
                    // Add capture button
                    const captureBtn = document.createElement('button');
                    captureBtn.textContent = 'Capture Frame';
                    captureBtn.onclick = () => captureFrame(video);
                    status.appendChild(captureBtn);
                }} else {{
                    throw new Error('No camera access method worked');
                }}
            }} catch (err) {{
                video.style.display = 'none';
                let errorMsg = '<div class="error">✗ ' + err.message + '</div>';
                
                if (err.name === 'NotAllowedError') {{
                    errorMsg += '<p>Camera permission was denied. Please grant permission and try again.</p>';
                }} else if (err.name === 'NotFoundError') {{
                    errorMsg += '<p>No camera found on this device.</p>';
                }} else if (err.name === 'NotReadableError') {{
                    errorMsg += '<p>Camera is being used by another application.</p>';
                }} else if (location.protocol === 'http:') {{
                    errorMsg += '<div class="instructions"><strong>HTTP Camera Access:</strong><br>';
                    errorMsg += '1. Use Chrome flags method above<br>';
                    errorMsg += '2. Or use the "Take Photo" button for file upload<br>';
                    errorMsg += '3. Or access via localhost instead of IP</div>';
                }}
                
                status.innerHTML = errorMsg;
            }}
        }}
        
        function handleFileUpload(event) {{
            const file = event.target.files[0];
            if (file) {{
                const reader = new FileReader();
                reader.onload = function(e) {{
                    const img = document.getElementById('capturedImage');
                    img.src = e.target.result;
                    img.style.display = 'block';
                    document.getElementById('status').innerHTML = '<div class="success">✓ Photo captured via file upload</div>';
                }};
                reader.readAsDataURL(file);
            }}
        }}
        
        function captureFrame(video) {{
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            const img = document.getElementById('capturedImage');
            img.src = canvas.toDataURL();
            img.style.display = 'block';
            
            document.getElementById('status').innerHTML += '<div class="success">✓ Frame captured!</div>';
        }}
        </script>
    </body>
    </html>
    '''

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

@app.route('/process_phone_frame', methods=['POST'])
def process_phone_frame():
    """Process a single frame from phone camera"""
    try:
        # Get the frame from the request
        frame_file = request.files.get('frame')
        if not frame_file:
            return jsonify({'error': 'No frame provided'}), 400
        
        # Convert to numpy array
        image_bytes = frame_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process the frame
        start_time = time.time()
        
        # Run inference on the frame
                # Run inference on the frame
        processed_frame, parking_stats = analyze_parking_spaces(frame)
        
        # Extract detection boxes if available
        detections = []
        
        end_time = time.time()
        
        # Calculate FPS
        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        # Get system metrics
        cpu, mem, gpu = get_system_metrics()
        
        # Prepare response
        response_data = {
            'occupied': parking_stats.get('occupied', 0),
            'empty': parking_stats.get('empty', 0),
            'fps': fps,
            'cpu': cpu,
            'mem': mem,
            'gpu': gpu,
            'detections': detections,
            'status': 'success'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error processing phone frame: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'occupied': 0,
            'empty': 0,
            'fps': 0
        }), 500

@app.route('/process_mobile_image', methods=['POST'])
def process_mobile_image():
    """Process image from mobile file upload"""
    try:
        image = request.files.get('image') or request.files.get('frame')
        if image:
            # Save temporarily
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mobile_capture.jpg')
            image.save(temp_path)
            
            # Process
            output_path, fps, parking_stats = run_inference(temp_path)
            
            # Get system metrics
            cpu, mem, gpu = get_system_metrics()
            
            return jsonify({
                'occupied': parking_stats.get('occupied', 0),
                'empty': parking_stats.get('empty', 0),
                'total': parking_stats.get('total', 0),
                'fps': fps,
                'cpu': cpu,
                'mem': mem,
                'gpu': gpu,
                'status': 'success'
            })
    except Exception as e:
        print(f"Error processing mobile image: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Add CORS headers for camera access
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    # Add permission policy for camera
    response.headers.add('Permissions-Policy', 'camera=*')
    return response

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
    # Get local IP address
    local_ip = get_local_ip()
    port = 5000
    
    print(f"\n{'='*60}")
    print(f"🚗 Parking Detection App Starting...")
    print(f"{'='*60}")
    print(f"📍 Local Access: http://127.0.0.1:{port}")
    print(f"🌐 Network Access: http://{local_ip}:{port}")
    print(f"{'='*60}")
    print(f"📱 Mobile Camera Access Instructions:")
    print(f"   1. Visit: http://{local_ip}:{port}/camera-test")
    print(f"   2. For Chrome Android: Enable insecure origins in chrome://flags")
    print(f"   3. Or use the file upload method (always works)")
    print(f"{'='*60}")
    print(f"🔧 Troubleshooting:")
    print(f"   - Camera test page: http://{local_ip}:{port}/camera-test")
    print(f"   - Make sure both devices are on same WiFi network")
    print(f"{'='*60}\n")
    
    # Configure Flask to run on all network interfaces
    app.run(
        host='0.0.0.0',  # Listen on all network interfaces
        port=port,
        debug=True,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent duplicate output
    )
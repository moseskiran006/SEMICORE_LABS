document.addEventListener('DOMContentLoaded', () => {
    const imageInput = document.getElementById('imageInput');
    const displayImage = document.getElementById('displayImage');
    const uploadedImage = document.getElementById('uploadedImage');
    const canvas = document.getElementById('detectionCanvas');
    const ctx = canvas.getContext('2d');
    const controlPanel = document.querySelector('.control-panel');

    imageInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (event) => {
            const base64Image = event.target.result;
            displayImage.src = base64Image;
            uploadedImage.style.display = 'block';

            // Process image on server
            const res = await fetch('/process_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: base64Image })
            });

            const data = await res.json();
            if (data.error) {
                console.error(data.error);
                return;
            }

            // Show updated image
            displayImage.src = data.image_base64;

            // Draw detections on canvas
            drawDetections(data.boxes);

            // Update metrics
            updateMetrics(data.metrics);

            // Add download link
            const existing = document.getElementById('downloadBtn');
            if (existing) existing.remove();

            const downloadBtn = document.createElement('a');
            downloadBtn.id = 'downloadBtn';
            downloadBtn.href = data.download_url;
            downloadBtn.textContent = "Download Annotated Image";
            downloadBtn.className = 'btn btn-success';
            downloadBtn.download = "annotated.jpg";
            controlPanel.appendChild(downloadBtn);
        };
        reader.readAsDataURL(file);
    });

    function drawDetections(boxes) {
        canvas.width = displayImage.width;
        canvas.height = displayImage.height;

        const scaleX = displayImage.width / displayImage.naturalWidth;
        const scaleY = displayImage.height / displayImage.naturalHeight;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        boxes.forEach(box => {
            const [x1, y1, x2, y2] = box.bbox;
            const color = box.class === 1 ? '#dc3545' : '#28a745';
            const label = box.label;

            const sx1 = x1 * scaleX;
            const sy1 = y1 * scaleY;
            const sx2 = x2 * scaleX;
            const sy2 = y2 * scaleY;

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);

            ctx.fillStyle = color;
            ctx.fillRect(sx1, sy1 - 20, 120, 20);
            ctx.fillStyle = 'white';
            ctx.fillText(`${label}: ${(box.confidence * 100).toFixed(1)}%`, sx1 + 5, sy1 - 5);
        });
    }

    function updateMetrics(metrics) {
        document.getElementById('fps').textContent = metrics.fps.toFixed(1);
        document.getElementById('inference').textContent = metrics.inference_time.toFixed(1);
        document.getElementById('cpu').textContent = metrics.cpu_usage.toFixed(1);
        document.getElementById('memory').textContent = metrics.memory_usage.toFixed(1);
        document.getElementById('occupiedSlots').textContent = metrics.occupied_slots;
        document.getElementById('emptySlots').textContent = metrics.empty_slots;
        document.getElementById('totalSlots').textContent = metrics.occupied_slots + metrics.empty_slots;
    }
});
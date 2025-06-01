from datetime import datetime  
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Constants
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'
EYE_THRESHOLD = 0.3
MIN_CONSECUTIVE_FRAMES = 2

# Verify cascade files exist
if not os.path.exists(CASCADE_PATH):
    error_msg = f"Face cascade file not found at {CASCADE_PATH}"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)

if not os.path.exists(EYE_CASCADE_PATH):
    error_msg = f"Eye cascade file not found at {EYE_CASCADE_PATH}"
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)

logger.info(f"Found face cascade at: {CASCADE_PATH}")
logger.info(f"Found eye cascade at: {EYE_CASCADE_PATH}")

# Load models
try:
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError("Failed to load face cascade classifier")
    
    eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
    if eye_cascade.empty():
        raise RuntimeError("Failed to load eye cascade classifier")
    
    logger.info("✅ OpenCV models loaded successfully!")
except Exception as e:
    error_msg = f"❌ Error loading OpenCV models: {e}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)

# Status tracking
sleep_frames = 0
drowsy_frames = 0
active_frames = 0

def detect_eyes(gray, face):
    """Detect eyes in the face region and determine if they're open."""
    (x, y, w, h) = face
    roi_gray = gray[y:y + h, x:x + w]
    
    # Detect eyes in the face region
    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # If no eyes are detected, consider them closed
    if len(eyes) == 0:
        return 0  # Closed
    
    # Calculate the relative size of detected eyes
    eye_sizes = [w * h for (x, y, w, h) in eyes]
    relative_eye_size = sum(eye_sizes) / (face[2] * face[3])
    
    if relative_eye_size < EYE_THRESHOLD:
        return 0  # Closed
    elif relative_eye_size < EYE_THRESHOLD * 1.5:
        return 1  # Drowsy
    else:
        return 2  # Open

def validate_image(file):
    """Validate the uploaded image file."""
    if not file:
        raise ValueError("No file uploaded")
    
    if file.filename == '':
        raise ValueError("No image selected")
    
    # Read image bytes
    image_bytes = file.read()
    if len(image_bytes) == 0:
        raise ValueError("Empty image file")
    
    return image_bytes

def process_image(image_bytes):
    """Process the image and detect drowsiness."""
    global sleep_frames, drowsy_frames, active_frames
    
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, "NO_FACE_DETECTED", None
        
        results = []
        for face in faces:
            # Get eye state
            eye_state = detect_eyes(gray, face)
            
            # Update status
            if eye_state == 0:  # Eyes closed
                sleep_frames += 1
                drowsy_frames = 0
                active_frames = 0
                status = "SLEEPING !!!"
                color = (0, 0, 255)  # Red
            elif eye_state == 1:  # Eyes drowsy
                drowsy_frames += 1
                sleep_frames = 0
                active_frames = 0
                if drowsy_frames > MIN_CONSECUTIVE_FRAMES:
                    status = "DIZZY"
                    color = (0, 255, 255)  # Yellow
                else:
                    status = "ACTIVE"
                    color = (0, 255, 0)  # Green
            else:  # Eyes open
                active_frames += 1
                sleep_frames = 0
                drowsy_frames = 0
                status = "ACTIVE"
                color = (0, 255, 0)  # Green
            
            # Draw annotations
            (x, y, w, h) = face
            cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Active Frames: {active_frames}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Detect and draw eyes
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            results.append({
                'status': status,
                'face_location': [int(x), int(y), int(w), int(h)],
                'eye_state': eye_state
            })
        
        # Convert frame to base64
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_base64, status, results
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return None, f"ERROR: {str(e)}", None

@app.route('/api/test', methods=['GET'])
def test():
    """Test endpoint to check server status."""
    return jsonify({
        'status': 'success',
        'message': 'Backend server is running!',
        'opencv_version': cv2.__version__,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/detect', methods=['POST'])
def detect():
    """Main endpoint for drowsiness detection."""
    try:
        # Validate request
        if 'image' not in request.files:
            print(request.files,"request.files")
            return jsonify({'error': 'No image uploaded'}), 400
            
        # Validate and get image bytes
        try:
            image_bytes = validate_image(request.files['image'])
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Process image
        img_base64, status, results = process_image(image_bytes)
        
        if img_base64 is None:
            return jsonify({'error': status}), 500
        
        response = {
            'image': img_base64,
            'status': status,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API documentation."""
    return jsonify({
        'message': 'Drowsiness Detection API',
        'version': '2.0.0',
        'endpoints': {
            'test': {
                'path': '/api/test',
                'method': 'GET',
                'description': 'Check server status'
            },
            'detect': {
                'path': '/api/detect',
                'method': 'POST',
                'description': 'Detect drowsiness in image',
                'params': {
                    'image': 'multipart/form-data image file'
                }
            }
        },
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\n🚀 Starting Drowsiness Detection Server...")
    print(f"📸 OpenCV version: {cv2.__version__}")
    print("🔍 Using Haar Cascade Classifiers for face and eye detection")
    print("🌐 Server will run on http://localhost:5001")
    print("🔄 CORS enabled for frontend integration\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
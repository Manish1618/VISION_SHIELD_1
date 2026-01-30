import os
import cv2
import torch
import timm
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from mtcnn import MTCNN

# -----------------------------
# Configuration
# -----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

MODEL_PATH = "deepfake_model.pth"
IMG_SIZE = 224

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Load Model
# -----------------------------
model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=1
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("Model loaded successfully")

# -----------------------------
# Face Detector
# -----------------------------
detector = MTCNN()

# -----------------------------
# Helper Functions
# -----------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]["box"]
    h_img, w_img, _ = rgb.shape
    
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    face = rgb[y:y+h, x:x+w]

    if face.size == 0:
        return None

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    return face

def predict_face(face):
    face = face.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    face = (face - mean) / std
    face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(face)
        prob = torch.sigmoid(pred)

    return prob.item()

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    preds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face = extract_face(frame)
        if face is not None:
            preds.append(predict_face(face))

    cap.release()

    if len(preds) == 0:
        return None

    # EMA smoothing
    alpha = 0.3
    ema = preds[0]
    for p in preds[1:]:
        ema = alpha * p + (1 - alpha) * ema

    return ema

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Determine file type
        ext = filename.rsplit('.', 1)[1].lower()
        
        if ext in ['jpg', 'jpeg', 'png']:
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({'error': 'Cannot read image'}), 400
            
            face = extract_face(img)
            if face is None:
                return jsonify({'error': 'No face detected in image'}), 400
            
            fake_prob = predict_face(face)
        else:
            fake_prob = predict_video(filepath)
            if fake_prob is None:
                return jsonify({'error': 'No face detected in video'}), 400

        # Clean up
        os.remove(filepath)

        real_prob = 1 - fake_prob
        
        return jsonify({
            'isFake': bool(fake_prob >= 0.5),
            'fakeConfidence': float(fake_prob * 100),
            'realConfidence': float(real_prob * 100)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

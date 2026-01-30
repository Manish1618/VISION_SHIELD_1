# VisionShield - Deepfake Detection System

A web-based AI-powered deepfake detection system using Flask and deep learning.

## Project Structure

```
hackathon/
├── app.py                    # Flask backend application
├── deepfake_model.pth        # Trained deepfake detection model
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── templates/
│   └── index.html           # Web interface
├── static/
│   ├── style.css            # Cyber-themed styling
│   └── script.js            # Frontend interaction logic
└── uploads/                 # Temporary file storage (auto-created)
```

## Installation & Setup

### 1. Create Virtual Environment
```bash
cd /Users/divyesh/Documents/hackathon
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

### 4. Access the Web Interface
Open your browser and navigate to:
```
http://localhost:8080
```

## Features

- **Drag & Drop Upload**: Upload images (JPG, PNG) or videos (MP4, AVI, MOV)
- **Real-time Detection**: Analyzes media using EfficientNet B0 model
- **Confidence Visualization**: Shows authenticity and manipulation probabilities
- **Cyber-themed UI**: Modern, responsive dark interface with animations
- **Face Detection**: Automatic face detection using MTCNN
- **Video Processing**: Processes video frames with EMA smoothing

## Supported File Types

- **Images**: JPG, JPEG, PNG
- **Videos**: MP4, AVI, MOV

## Model Details

- **Architecture**: EfficientNet B0
- **Input Size**: 224x224 pixels
- **Face Detector**: MTCNN
- **Output**: Binary classification (Real/Fake)
- **Smoothing**: Exponential Moving Average (EMA) for video frames

## Dependencies

- Flask 3.0.0
- PyTorch 2.1.0
- TorchVision 0.16.0
- TIMM 0.9.12
- OpenCV 4.8.1.78
- MTCNN 0.1.1
- TensorFlow 2.20.0
- NumPy <2
- Pillow 10.1.0
- Werkzeug 3.0.1

## Important Notes

- Model file (`deepfake_model.pth`) must be present in the root directory
- Uploads folder is created automatically on first run
- Maximum file size: 50MB
- CPU/GPU detection is automatic

## Troubleshooting

### NumPy Compatibility Issues
If you encounter NumPy 2.0 compatibility errors, the requirements are pinned to `numpy<2` which resolves OpenCV and other package conflicts.

### TensorFlow Installation
TensorFlow is required by MTCNN for face detection. If installation fails, ensure you have sufficient disk space and internet connection.

### Model Not Found Error
Ensure `deepfake_model.pth` is in the project root directory before running the app.

## License

VisionShield Team - SECURE :: DETECT :: PROTECT

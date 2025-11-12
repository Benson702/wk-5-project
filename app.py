import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    net = models.resnet18(pretrained=False)
    num_features = net.fc.in_features
    num_classes = 7  # adjust to match your training
    net.fc = nn.Linear(num_features, num_classes)
    
    net.load_state_dict(torch.load("emotion_model.pth", map_location="cpu"))
    net.eval()
    return net
model = load_model()
class_names = ['angry', 'happy', 'sad', 'surprise', 'neutral', 'fear', 'disgust']  # adjust as needed

# -------------------------------
# Define preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
FRAME_WINDOW = st.image([])

# Determine if cv2 imported successfully
try:
    # Use getattr to avoid static analyzers complaining about missing __version__ attribute
    _ = getattr(cv2, "__version__", None)
    cv2_available = True
except Exception:
    cv2_available = False
if not cv2_available:
    st.error("OpenCV (cv2) is not installed or could not be imported. Install it with: pip install opencv-python. Webcam features are disabled.")
else:
    # Use getattr to access VideoCapture to avoid static analysis errors when cv2 stubs are incomplete
    from typing import Any
    VideoCapture = getattr(cv2, "VideoCapture", None)
    if VideoCapture is None:
        st.error("cv2.VideoCapture is not available in this environment; webcam features are disabled.")
    else:
        camera: Any = VideoCapture(0)
        run = True

        while run:
            if not getattr(camera, "isOpened", lambda: False)():
                st.warning("Camera not found or not accessible.")
                break

            ret, frame = camera.read()
            if not ret:
                st.warning("Failed to read frame from camera.")
                break

            # Convert BGR (OpenCV) to RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            
            # Preprocess the image
            input_tensor = transform(pil_img).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)
                emotion = class_names[preds.item()]

            # Add label on image
            cv2.putText(img, f'Emotion: {emotion}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            FRAME_WINDOW.image(img)

        # Release the camera if it was created
        try:
            camera.release()
        except Exception:
            pass

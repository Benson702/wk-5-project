😄 Facial Emotion Recognition App
📘 Overview
This project is an AI-powered facial emotion recognition system that detects human emotions from facial images or live webcam feeds.
It leverages Deep Learning (ResNet-18) and a custom Facial Emotion Recognition Dataset (with images like angry, happy, sad, neutral, etc.) to classify emotions accurately.
The app provides a simple, interactive web interface built using Streamlit, making it easy for anyone to upload an image or use their camera to analyze facial emotions in real time.

🎯 Objectives


Build a deep learning model to classify facial emotions accurately.


Develop a user-friendly web interface for image or camera input.


Demonstrate a complete AI Development Workflow — from dataset preparation to model deployment.



👥 Stakeholders


Developers / Data Scientists — for model training and experimentation.


Educators / Therapists — to understand emotional patterns in learning or therapy contexts.


Human-Computer Interaction Designers — for emotion-aware user interfaces.



🚀 Features


🔍 Upload an image to detect the emotion shown.


🎥 Use your webcam for real-time emotion recognition.


📊 Display of model prediction probabilities.


🧠 Built with PyTorch + ResNet18, fine-tuned on a facial emotion dataset.


🌐 Streamlit app for interactive visualization and deployment.






🧠 Model Details


Base Model: ResNet-18 (pretrained on ImageNet).


Fine-Tuning: Last layer modified for your dataset’s emotion classes (e.g., 5 or 7).


Input Size: 96×96 grayscale images.


Loss Function: CrossEntropyLoss.


Optimizer: Adam / SGD.



📊 Emotion Classes
LabelEmotion Name0Angry 😠1Happy 😄2Sad 😢3Neutral 😐4Surprise 😮(optional)Fear 😨(optional)Disgust 🤢


⚖️ Ethical Considerations


Ensure data diversity to avoid bias across age, gender, or ethnicity.


Use responsibly — emotion recognition is probabilistic and may misclassify.



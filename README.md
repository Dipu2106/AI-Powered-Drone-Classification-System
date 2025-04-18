# AI-Powered-Drone-Classification-System
This project is a final-year Bachelor’s Engineering project focused on classifying aerial objects—such as birds or drones—using spectrogram data derived from micro-Doppler radar signatures. The core objective is to train a deep learning model that can distinguish between different object classes based on their spectrograms.

# 📖 Project Summary
This project presents a deep learning system that classifies aerial objects—specifically birds and drones—by analyzing motion signals derived from video footage. Using optical flow techniques, we extract 1D motion patterns that reflect the unique movement characteristics of each object type. These patterns are then fed into a 1D Convolutional Neural Network (CNN) to enable accurate classification.

The system supports real-time inference through a Flask-based web interface and has practical use cases in airspace security, wildlife monitoring, and automated surveillance.

# ⚙️ Core Components
### 🌀 Optical Flow Signal Extraction
Frames are extracted from videos using OpenCV.

Farneback's method is used to compute dense optical flow.

The flow vectors are converted into 1D motion signals for downstream analysis.

### 🧪 Dataset Generation
From 161 original video files, we generated a synthetic dataset with 500 samples (250 drone, 250 bird).

All signals are normalized and padded to ensure consistent input size for training.

### 🧠 1D CNN Classification Model
A custom 1D Convolutional Neural Network is designed to process temporal motion signals.

The model achieved an overall classification accuracy of 77%.

### 🌐 Web-Based Inference
A lightweight Flask web app allows users to upload videos.

The app processes the video, classifies the object, and returns the result along with a visualization of the signal.

### 🧱 System Workflow
User uploads a video file via the web interface.

Optical flow signal is computed from the video frames.

The 1D CNN model classifies the motion pattern.

Output is returned to the user with visual feedback.

# 📈 Performance Highlights
✅ Accuracy: 77% on the synthetic test set

⏱ Average Inference Time: 20–30 seconds per video

🧠 Model Type: 1D CNN (Keras + TensorFlow)

# 💡 Use Cases
Restricted Airspace Monitoring – Detect unauthorized drones and distinguish them from birds.

Conservation Research – Monitor bird activity in protected habitats without invasive methods.

Smart Surveillance – Integrate intelligent object classification into camera networks.

# 🔮 Future Enhancements
Increase data volume and diversity for better generalization.

Investigate hybrid architectures like CNN-LSTM or Vision Transformers.

Optimize processing for edge deployment (e.g., Jetson Nano, Raspberry Pi).

Introduce multi-class support for additional aerial entities.

Explore fusion with non-visual sensors (e.g., radar, acoustic).


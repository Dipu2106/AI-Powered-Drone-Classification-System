import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (ensure the path to your model is correct)
model = load_model('micro_doppler_model.keras')

# Directory to save uploaded videos and spectrogram images
UPLOAD_FOLDER = 'uploads'
SPECTROGRAM_FOLDER = 'spectrograms'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SPECTROGRAM_FOLDER'] = SPECTROGRAM_FOLDER

# Ensure upload and spectrogram folders exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(SPECTROGRAM_FOLDER):
    os.makedirs(SPECTROGRAM_FOLDER)

# Function to extract frames from video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Function to calculate optical flow and return a 1D signal
def calculate_optical_flow(frames):
    flow_data = []
    prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_data.append(mag.mean())  # Aggregate to create 1D signal
        prev_frame = curr_frame
    return flow_data

# Function to generate a spectrogram from the flow signal and save it as an image
def generate_spectrogram(flow_signal, filename='spectrogram.png'):
    # Create the spectrogram
    plt.figure(figsize=(10, 5))
    plt.specgram(flow_signal, NFFT=256, Fs=2, noverlap=128, scale='dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    
    # Save the spectrogram as an image
    spectrogram_path = os.path.join(SPECTROGRAM_FOLDER, filename)
    plt.savefig(spectrogram_path)
    plt.close()

    return spectrogram_path

# Function to preprocess the flow signal and make predictions
def preprocess_and_predict(video_path):
    frames = extract_frames(video_path)
    flow_signal = calculate_optical_flow(frames)
    
    # Pad or truncate flow_signal to a fixed length
    target_length = 100
    flow_signal = np.pad(flow_signal, (0, target_length - len(flow_signal)), 'constant') if len(flow_signal) < target_length else flow_signal[:target_length]
    
    # Convert flow_signal to a numpy array and reshape for the model (Conv1D expects 3D input)
    flow_signal = np.array(flow_signal).reshape(1, target_length, 1)
    
    # Make a prediction using the trained model
    prediction = model.predict(flow_signal)
    result = 'Drone' if prediction[0] > 0.5 else 'Bird'  # Assuming binary classification: Drone = 1, Bird = 0
    
    return result, flow_signal

# Route to serve spectrogram images
@app.route('/spectrograms/<filename>')
def serve_spectrogram(filename):
    return send_from_directory(SPECTROGRAM_FOLDER, filename)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle video upload, prediction, and spectrogram generation
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded video to the server
    video_filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    file.save(video_path)
    
    # Process the video, calculate flow signal, and make prediction
    result, flow_signal = preprocess_and_predict(video_path)
    
    # Generate spectrogram and save the image
    spectrogram_filename = f"spectrogram_{video_filename}.png"
    spectrogram_path = generate_spectrogram(flow_signal[0], spectrogram_filename)
    
    # Return the result and spectrogram image path
    spectrogram_url = f'/spectrograms/{spectrogram_filename}'
    return jsonify({'prediction': result, 'spectrogram': spectrogram_url})

if __name__ == '__main__':
    app.run(debug=True)

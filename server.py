import os
import time
import numpy as np
import librosa
from flask import Flask, request
import joblib

app = Flask(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(script_directory, "uploads")
MODEL_PATH = os.path.join(script_directory, "rf_model.pkl")
AUDIO_FILE = os.path.join(UPLOAD_FOLDER, "recorded_audio.wav")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = joblib.load(MODEL_PATH)


def wav_to_mfcc(audio_data, sr, n_mfcc=20, max_len=400, fmin=0, fmax=None):
    duration = len(audio_data) / sr
    end_sample = int(sr * duration)
    y_segment = audio_data[:end_sample]

    # Normalize the audio signal
    y_segment = librosa.util.normalize(y_segment)

    # Compute the Mel spectrogram with the specified frequency range
    mel_spectrogram = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=128, fmin=fmin, fmax=fmax)

    # Convert the Mel spectrogram to MFCCs
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


def flatten_mfcc(mfcc):
    return mfcc.flatten()


def predict_environment_sound(audio_file, n_mfcc=20, max_len=400):
    audio_data, sr = librosa.load(audio_file, sr=None)

    mfcc = wav_to_mfcc(audio_data, sr, n_mfcc=n_mfcc, max_len=max_len, fmin=250, fmax=3250)

    flattened_mfcc = flatten_mfcc(mfcc).reshape(1, -1)

    prediction = model.predict(flattened_mfcc)

    predicted_class = prediction[0]

    if predicted_class == 1:
        return 'Alarm'
    elif predicted_class == 2:
        return 'Baby'
    elif predicted_class == 3:
        return 'Dog'
    else:
        return 'Unknown'


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Perform prediction
        result = predict_environment_sound(filepath)
        print(f"Predicted class for the latest file => {result}")
        #return f"File uploaded successfully: {filepath}. Predicted class: {result}", 200
        return f"File uploaded successfully: {filepath}. Predicted class: {result}", 200


if __name__ == '__main__':
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000)

import os
import torch
import librosa
import numpy as np
from src.model import GRUEmotionClassifier
from src.predict import predict_file


MODEL_PATH = "emotion_gru.pth"
AUDIO_PATH = "//Users/pedroh/Desktop/ToneAI/1011.wav"  # path to your file


def print_audio_details(path):
    y, sr = librosa.load(path, sr=None, mono=True)

    print("\n AUDIO FILE DETAILS")
    print("------------------------------------------")
    print(f"File: {os.path.basename(path)}")
    print(f"Sample Rate: {sr} Hz")
    print(f"Samples: {len(y)}")
    print(f"Duration: {len(y)/sr:.3f} seconds")
    print(f"Min Amplitude: {np.min(y):.4f}")
    print(f"Max Amplitude: {np.max(y):.4f}")
    print(f"Mean Amplitude: {np.mean(y):.4f}")
    print(f"Std Amplitude: {np.std(y):.4f}")
    print("------------------------------------------")


def main():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    # Check if audio exists
    if not os.path.exists(AUDIO_PATH):
        print(f"Audio file not found: {AUDIO_PATH}")
        return

    # Print audio details
    print_audio_details(AUDIO_PATH)

    # Load model
    device = torch.device("cpu")
    model = GRUEmotionClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Predict
    label, _ = predict_file(AUDIO_PATH, model, device)

    print("\n PREDICTION RESULT")
    print("------------------------------------------")
    print(f"Emotion Detected: {label}")
    print("------------------------------------------")


if __name__ == "__main__":
    main()
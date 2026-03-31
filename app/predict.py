import numpy as np
import librosa
import cv2

from model_loader import base_model, feature_model, prototypes, class_names, OSR_THRESHOLD

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, duration=4.0)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_norm = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    mel_rgb = cv2.applyColorMap(mel_norm, cv2.COLORMAP_VIRIDIS)
    img = cv2.resize(mel_rgb, (224, 224))

    img = np.expand_dims(img, axis=0)
    return img


def predict_sound(file_path):
    img = preprocess_audio(file_path)

    # 🔹 Feature extraction
    features = feature_model.predict(img, verbose=0)[0]

    # 🔹 Distance to prototypes
    distances = [np.linalg.norm(features - prototypes[i]) for i in range(len(class_names))]

    closest_idx = np.argmin(distances)
    min_dist = distances[closest_idx]

    if min_dist > OSR_THRESHOLD:
        return {
            "label": "unknown",
            "confidence": float(min_dist)
        }

    return {
        "label": class_names[closest_idx],
        "confidence": float(min_dist)
    }

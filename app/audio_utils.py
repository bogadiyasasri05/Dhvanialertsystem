import librosa
import numpy as np
import cv2

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, duration=4.0)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_norm = cv2.normalize(mel_spec_db, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    mel_rgb = cv2.applyColorMap(mel_norm, cv2.COLORMAP_VIRIDIS)
    img = cv2.resize(mel_rgb, (224, 224))

    img = np.expand_dims(img, axis=0)
    return img

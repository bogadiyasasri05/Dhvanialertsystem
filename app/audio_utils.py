import librosa
import numpy as np
import cv2

def split_audio(file_path, chunk_duration=4.0):
    y, sr = librosa.load(file_path)
    chunk_size = int(chunk_duration * sr)

    chunks = []
    for i in range(0, len(y), chunk_size):
        chunk = y[i:i+chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)

    return chunks, sr


def chunk_to_image(chunk, sr):
    mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_norm = cv2.normalize(
        mel_db, None, 0, 255,
        cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    mel_rgb = cv2.applyColorMap(mel_norm, cv2.COLORMAP_VIRIDIS)
    img = cv2.resize(mel_rgb, (224, 224))

    return img
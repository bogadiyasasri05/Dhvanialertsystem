import numpy as np
import librosa
import soundfile as sf
import io

def audio_to_spectrogram(file_bytes):
    # Read audio
    data, sr = sf.read(io.BytesIO(file_bytes))

    # Convert stereo → mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Generate mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=data, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram)

    # Resize to 224x224
    spectrogram = np.resize(spectrogram, (224, 224))

    # Convert to 3 channels
    spectrogram = np.stack([spectrogram]*3, axis=-1)

    # Normalize
    spectrogram = spectrogram / (np.max(spectrogram) + 1e-6)

    return np.expand_dims(spectrogram, axis=0)

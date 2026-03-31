import numpy as np
from app.audio_utils import preprocess_audio
from app.model_loader import feature_model, prototypes, class_names, OSR_THRESHOLD

def predict_sound(file_path):
    img = preprocess_audio(file_path)

    features = feature_model.predict(img, verbose=0)[0]

    distances = [
        np.linalg.norm(features - prototypes[i])
        for i in range(len(class_names))
    ]

    closest_idx = int(np.argmin(distances))
    min_dist = float(distances[closest_idx])

    if min_dist > OSR_THRESHOLD:
        return {
            "label": "unknown",
            "confidence": min_dist
        }

    return {
        "label": class_names[closest_idx],
        "confidence": min_dist
    }

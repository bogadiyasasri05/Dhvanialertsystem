import numpy as np
from app.model_loader import load_model_once, feature_model, prototypes, class_names, OSR_THRESHOLD

def predict_sound(image):
    load_model_once()  # ✅ load only when needed

    # normalize
    image = image / 255.0

    features = feature_model.predict(image)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    proto_norm = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)

    similarity = np.dot(features, proto_norm.T)

    best_idx = np.argmax(similarity)
    best_score = similarity[0][best_idx]

    if best_score < OSR_THRESHOLD:
        return {"class": "unknown", "confidence": float(best_score)}

    return {
        "class": class_names[best_idx],
        "confidence": float(best_score)
    }

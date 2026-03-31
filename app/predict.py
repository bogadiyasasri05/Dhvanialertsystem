import numpy as np
from app.model_loader import feature_model, prototypes, class_names, OSR_THRESHOLD

def predict_sound(image):
    # Step 1: Extract features
    features = feature_model.predict(image)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # Step 2: Normalize prototypes
    proto_norm = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)

    # Step 3: Cosine similarity
    similarity = np.dot(features, proto_norm.T)

    # Step 4: Get best match
    best_idx = np.argmax(similarity)
    best_score = similarity[0][best_idx]

    # Step 5: Open Set Recognition
    if best_score < OSR_THRESHOLD:
        return {
            "class": "unknown",
            "confidence": float(best_score)
        }

    return {
        "class": class_names[best_idx],   # ✅ THIS gives class name
        "confidence": float(best_score)
    }

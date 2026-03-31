import numpy as np
from .model_loader import feature_model, prototypes, class_names, OSR_THRESHOLD
from .audio_utils import split_audio, chunk_to_image


def predict_chunk(img):
    img = np.expand_dims(img, axis=0)
    features = feature_model.predict(img, verbose=0)[0]

    distances = [
        np.linalg.norm(features - prototypes[i])
        for i in range(len(class_names))
    ]

    closest_idx = np.argmin(distances)
    min_dist = distances[closest_idx]

    if min_dist > OSR_THRESHOLD:
        return {
            "status": "unknown",
            "closest_match": class_names[closest_idx],
            "distance": float(min_dist)
        }
    else:
        return {
            "status": "known",
            "class": class_names[closest_idx],
            "distance": float(min_dist)
        }


def predict_full_audio(file_path):
    chunks, sr = split_audio(file_path)

    all_results = []

    for chunk in chunks:
        img = chunk_to_image(chunk, sr)
        result = predict_chunk(img)
        all_results.append(result)

    # 🔥 Final decision logic
    known_results = [r for r in all_results if r["status"] == "known"]

    if known_results:
        best = min(known_results, key=lambda x: x["distance"])
        return {
            "final_status": "known",
            "prediction": best,
            "chunks_analyzed": len(all_results)
        }
    else:
        best = min(all_results, key=lambda x: x["distance"])
        return {
            "final_status": "unknown",
            "prediction": best,
            "chunks_analyzed": len(all_results)
        }
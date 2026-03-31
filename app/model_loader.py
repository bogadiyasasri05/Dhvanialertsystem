import tensorflow as tf
import numpy as np

MODEL_PATH = "app/models/dhvani_model_clean.h5"
PROTO_PATH = "app/models/class_prototypes.npy"

feature_model = None
prototypes = None
class_names = None
OSR_THRESHOLD = 0.5

def load_model_once():
    global feature_model, prototypes, class_names

    if feature_model is None:
        print("🔄 Loading model...")

        feature_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        prototypes = np.load(PROTO_PATH, allow_pickle=True)

        if isinstance(prototypes, dict):
            class_names = list(prototypes.keys())
            prototypes = np.array(list(prototypes.values()))
        else:
            class_names = [f"class_{i}" for i in range(len(prototypes))]

        print("✅ Model loaded successfully")

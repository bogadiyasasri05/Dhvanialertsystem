import tensorflow as tf
import numpy as np

MODEL_PATH = "app/models/dhvani_model_clean.h5"
PROTO_PATH = "app/models/class_prototypes.npy"

# ✅ Load FULL trained model (NO manual architecture)
feature_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ✅ Load prototypes
prototypes = np.load(PROTO_PATH, allow_pickle=True)

# ✅ Extract class names
if isinstance(prototypes, dict):
    class_names = list(prototypes.keys())
    prototypes = np.array(list(prototypes.values()))
else:
    class_names = [f"class_{i}" for i in range(len(prototypes))]

OSR_THRESHOLD = 0.5

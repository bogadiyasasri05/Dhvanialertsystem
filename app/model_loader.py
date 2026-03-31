import tensorflow as tf
import numpy as np

MODEL_PATH = "models/dhvani_model.h5"
PROTO_PATH = "models/class_prototypes.npy"

# ✅ Rebuild architecture (MobileNetV2-based)
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dense(128, activation="relu")(x)

feature_model = tf.keras.Model(inputs=base.input, outputs=x)

# ✅ Load weights only (avoids ALL keras errors)
feature_model.load_weights(MODEL_PATH)

# Load prototypes
prototypes = np.load(PROTO_PATH)

# ⚠️ UPDATE THESE
class_names = ["ambulance", "firetruck", "police"]  # <-- your classes
OSR_THRESHOLD = 0.5

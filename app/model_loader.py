import tensorflow as tf
import numpy as np

MODEL_PATH = "models/dhvani_model.h5"
PROTO_PATH = "models/class_prototypes.npy"

# Build model architecture
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dense(128, activation="relu")(x)

feature_model = tf.keras.Model(inputs=base.input, outputs=x)

# Load weights
feature_model.load_weights(MODEL_PATH)

# Load prototypes
prototypes = np.load(PROTO_PATH, allow_pickle=True)

# Extract class names
if isinstance(prototypes, dict):
    class_names = list(prototypes.keys())
    prototypes = np.array(list(prototypes.values()))
else:
    class_names = [f"class_{i}" for i in range(len(prototypes))]

OSR_THRESHOLD = 0.5

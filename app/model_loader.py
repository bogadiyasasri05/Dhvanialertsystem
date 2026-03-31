import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model

MODEL_PATH = "models/dhvani_model.keras"
PROTO_PATH = "models/class_prototypes.npy"

# Load model
base_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded")

# Feature extractor
feature_model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer("feature_extractor").output
)
print("✅ Feature extractor ready")

# Load prototypes
prototypes = np.load(PROTO_PATH, allow_pickle=True).item()
print("✅ Prototypes loaded")

class_names = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot',
    'jackhammer', 'siren', 'street_music'
]

OSR_THRESHOLD = 8.5

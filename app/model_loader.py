import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model

MODEL_PATH = "models/Emergency_Audio_MobileNet_FIXED.h5"
PROTO_PATH = "models/class_prototypes.npy"

# Load model safely
base_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("✅ Model loaded")

# If this layer name fails, we’ll adjust later
feature_model = Model(
    inputs=base_model.input,
    outputs=base_model.layers[-2].output   # 🔥 SAFE fallback
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

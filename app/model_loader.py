import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model

MODEL_PATH = "models/Emergency_Audio_MobileNet_FIXED1.h5"
PROTO_PATH = "models/class_prototypes.npy"

# ✅ Load model safely
try:
    base_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", str(e))
    raise e


# ✅ Feature extractor (safe fallback)
try:
    feature_model = Model(
        inputs=base_model.input,
        outputs=base_model.layers[-2].output   # fallback layer
    )
    print("✅ Feature extractor ready")
except Exception as e:
    print("❌ Feature extractor error:", str(e))
    raise e


# ✅ Load prototypes
try:
    prototypes = np.load(PROTO_PATH, allow_pickle=True).item()
    print("✅ Prototypes loaded")
except Exception as e:
    print("❌ Error loading prototypes:", str(e))
    raise e


# ✅ Class labels
class_names = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
    'drilling', 'engine_idling', 'gun_shot',
    'jackhammer', 'siren', 'street_music'
]

# ✅ OSR Threshold
OSR_THRESHOLD = 8.5

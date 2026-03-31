from fastapi import FastAPI, UploadFile, File
import shutil
import os
import time

app = FastAPI()

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ SAFE IMPORT (prevents app crash)
try:
    from app.predict import predict_full_audio
    MODEL_LOADED = True
    print("✅ Model + predict loaded successfully")
except Exception as e:
    MODEL_LOADED = False
    ERROR_MSG = str(e)
    print("❌ Error loading predict module:", ERROR_MSG)


@app.get("/")
def home():
    return {
        "message": "Dhvani AI running 🚀",
        "model_loaded": MODEL_LOADED
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # ❌ If model failed, return error instead of crashing
    if not MODEL_LOADED:
        return {
            "error": "Model not loaded",
            "details": ERROR_MSG
        }

    start_time = time.time()

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # ✅ Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ✅ Run prediction
        result = predict_full_audio(file_path)

        end_time = time.time()
        result["processing_time_sec"] = round(end_time - start_time, 2)

        return result

    except Exception as e:
        return {
            "error": "Prediction failed",
            "details": str(e)
        }

    finally:
        # ✅ Clean up file after prediction
        if os.path.exists(file_path):
            os.remove(file_path)

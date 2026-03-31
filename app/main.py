from fastapi import FastAPI, UploadFile, File
import shutil
import os
import time

from app.predict import predict_full_audio

app = FastAPI()

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Dhvani AI running 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predict full audio
    result = predict_full_audio(file_path)

    end_time = time.time()
    result["processing_time_sec"] = round(end_time - start_time, 2)

    return result
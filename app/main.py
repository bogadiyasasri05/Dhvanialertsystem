from fastapi import FastAPI, UploadFile, File
from app.predict import predict_sound
from app.audio_utils import audio_to_spectrogram

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Dhvani Audio API running 🎧"}

@app.get("/test")
def test():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert audio → spectrogram
    image = audio_to_spectrogram(contents)

    result = predict_sound(image)
    return result

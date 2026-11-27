"""
FastAPI app serving the Egg Hunt parasite-egg classifier.
"""
from fastapi import FastAPI, UploadFile

from egg_hunt.config import MODEL_PATH
from egg_hunt.predict import load_model, predict

app = FastAPI()

# Load the model once at startup, not on every request
app.state.model = load_model(MODEL_PATH)


@app.get("/")
def root():
    return {"message": "Welcome to Egg Hunt!"}


@app.post("/predict")
async def predict_endpoint(img: UploadFile, top_k: int = 3):
    """Upload a cropped egg image -> top_k species with confidence."""
    image_bytes = await img.read()
    result = predict(app.state.model, image_bytes, top_k=top_k)[0]
    return {"predictions": result}

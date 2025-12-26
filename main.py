from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile

from lung_pipeline import predict_lung_sound

app = FastAPI(title="Lung Sound Disease Classifier")

# Optional: allow frontends from other origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Lung sound classifier API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept a WAV file upload and return the predicted label."""
    # Save uploaded file to a temporary WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    result = predict_lung_sound(tmp_path)
    return {
        "filename": file.filename,
        "prediction": result["label"],
        "band_energy": result["band_energy"],
    }

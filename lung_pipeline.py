import joblib
import numpy as np
import librosa

MODEL_PATH = "lung_model.joblib"

# Load trained model + label encoder once when this module is imported
_artifact = joblib.load(MODEL_PATH)
_model = _artifact["model"]
_label_encoder = _artifact["label_encoder"]


def compute_band_energy_from_wav(path: str, low_f: float = 100.0, high_f: float = 2000.0) -> float:
    """Compute FFT-based band energy for a full WAV file, matching the training feature."""
    y_clip, sr = librosa.load(path, sr=None)
    Y = np.fft.rfft(y_clip)
    freqs = np.fft.rfftfreq(len(y_clip), d=1.0 / sr)
    mag = np.abs(Y)
    band_mask = (freqs >= low_f) & (freqs <= high_f)
    band_energy = float(np.sum(mag[band_mask] ** 2))
    return band_energy


def predict_lung_sound(path: str) -> dict:
    """Given a WAV file path, return the predicted label and band energy."""
    band_energy = compute_band_energy_from_wav(path, low_f=100.0, high_f=2000.0)
    X = np.array([[band_energy]], dtype=float)
    pred_idx = _model.predict(X)[0]
    label = _label_encoder.inverse_transform([pred_idx])[0]
    return {
        "label": label,
        "band_energy": band_energy,
    }

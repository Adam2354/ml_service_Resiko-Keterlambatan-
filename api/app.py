from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(
    title="SIGMA ML API",
    version="1.0.0",
    description="API klasifikasi risiko keterlambatan skripsi menggunakan Random Forest"
)

# ======================
# Load Model
# ======================

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "sigma_random_forest.pkl"
)

label_encoder = joblib.load(
    BASE_DIR / "models" / "label_encoder.pkl"
)

# ======================
# Request Schema
# ======================

class PredictionRequest(BaseModel):
    progres_skripsi: float
    frekuensi_bimbingan_bulanan: int
    keterlambatan_milestone_hari: int
    streak_aktivitas_mingguan: int
    jumlah_revisi_aktif: int

# ======================
# Routes
# ======================

@app.get("/")
def root():
    return {
        "message": "SIGMA Random Forest API Running"
    }

@app.post("/predict")
def predict(data: PredictionRequest):

    df = pd.DataFrame([{
        "progres_skripsi": data.progres_skripsi,
        "frekuensi_bimbingan_bulanan": data.frekuensi_bimbingan_bulanan,
        "keterlambatan_milestone_hari": data.keterlambatan_milestone_hari,
        "streak_aktivitas_mingguan": data.streak_aktivitas_mingguan,
        "jumlah_revisi_aktif": data.jumlah_revisi_aktif
    }])

    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    label = label_encoder.inverse_transform([pred])[0]

    confidence = float(max(proba))

    probabilities = {
        label_encoder.classes_[i]: round(float(proba[i]), 4)
        for i in range(len(label_encoder.classes_))
    }

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "probabilities": probabilities
    }
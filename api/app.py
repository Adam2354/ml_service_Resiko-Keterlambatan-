from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(
    title="SIGMA ML API"
)

# Load model
model = joblib.load("../models/sigma_random_forest.pkl")
label_encoder = joblib.load("../models/label_encoder.pkl")


@app.get("/")
def root():
    return {
        "message": "SIGMA Random Forest API Running"
    }


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([{
        "progres_skripsi": data["progres_skripsi"],
        "frekuensi_bimbingan_bulanan": data["frekuensi_bimbingan_bulanan"],
        "keterlambatan_milestone_hari": data["keterlambatan_milestone_hari"],
        "streak_aktivitas_mingguan": data["streak_aktivitas_mingguan"],
        "jumlah_revisi_aktif": data["jumlah_revisi_aktif"]
    }])

    pred = model.predict(df)[0]

    proba = model.predict_proba(df)[0]

    label = label_encoder.inverse_transform([pred])[0]

    confidence = float(max(proba))

    classes = label_encoder.classes_

    probabilities = {
        classes[i]: round(float(proba[i]), 4)
        for i in range(len(classes))
    }

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "probabilities": probabilities
    }
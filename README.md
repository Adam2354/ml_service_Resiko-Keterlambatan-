# SIGMA ML Service

Machine Learning Service untuk klasifikasi risiko keterlambatan skripsi mahasiswa menggunakan Random Forest.

## Dataset

300 data sintetis mahasiswa skripsi.

## Model

Random Forest Classifier

## Hasil Evaluasi

Accuracy : 96.67%

F1 Macro : 96.61%

Recall Berisiko : 100%

## Menjalankan API

uvicorn api.app:app --reload
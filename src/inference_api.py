import joblib
import numpy as np
from fastapi import FastAPI

app = FastAPI()
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.post("/predict")
def predict(features: list):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    pred = model.predict_proba(features)[0][1]
    return {"prediction": float(pred)}

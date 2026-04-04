"""API FastAPI pour modèle Heart Disease"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import mlops_tp.config as config

# Schéma Pydantic pour /predict
class PredictionRequest(BaseModel):
    features: dict

# Charger modèle AU DÉMARRAGE
model = None
try:
    model = joblib.load(config.MODEL_PATH)
    print(" Modèle chargé au démarrage")
except Exception as e:
    print(f" Erreur chargement modèle : {e}")

app = FastAPI(title="Heart Disease API", version=config.MODEL_VERSION)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None, "version": "1.0.1"}

@app.get("/metadata")
async def get_metadata():
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    return {
        "model_version": config.MODEL_VERSION,
        "task": "classification",
        "features": feature_names,
        "classes": ["no_disease", "disease"]
    }

@app.get("/data/sample")
async def get_data_sample(n: int = 5):
    try:
        df = pd.read_csv(config.DATA_PATH)
        return JSONResponse(df.head(n).to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur données : {str(e)}")

@app.post("/predict")
async def predict_endpoint(request: PredictionRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    start_time = time.time()
    
    try:
        # Préprocesser comme dans train.py
        features_df = pd.DataFrame([request.features])
        
        # Prédiction
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "prediction": int(prediction),
            "proba": {
                "no_disease": float(probabilities[0]),
                "disease": float(probabilities[1])
            },
            "task": "classification",
            "model_version": config.MODEL_VERSION,
            "latency_ms": round(latency, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur prédiction : {str(e)}")

"""Fonctions d'inférence"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
import mlops_tp.config as config
from pathlib import Path

def load_model():
    model_path = Path(config.MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle manquant : {model_path}")
    return joblib.load(model_path)

def predict(features: dict) -> dict:
    """Prédiction Heart Disease"""
    model = load_model()
    
    # FIX : Créer DataFrame avec NOMS de colonnes (pas array brut)
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # DataFrame avec noms colonnes exacts
    df = pd.DataFrame([features], columns=feature_names)
    
    # Prédire
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    
    return {
        'prediction': int(prediction),
        'proba': {
            'no_disease': float(proba[0]),
            'disease': float(proba[1])
        }
    }

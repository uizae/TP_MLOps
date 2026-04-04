"""Schemas Pydantic pour validation API"""
from pydantic import BaseModel
from typing import Dict, Any

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: int
    proba: Dict[str, float]
    task: str = "classification"
    model_version: str
    latency_ms: float

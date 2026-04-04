"""Entraînement Heart Disease avec MLflow"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import mlops_tp.config as config
import time

mlflow.set_experiment("heart_disease_mlops")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Docker MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("heart_disease_mlops")

    logger.info("Chargement des données")
    df = pd.read_csv(config.DATA_PATH)
    logger.info(f"Dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()
    
    # Split train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SPLIT, random_state=config.RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VAL_SPLIT/(1-config.TEST_SPLIT), 
        random_state=config.RANDOM_STATE, stratify=y_temp
    )
    
    logger.info(f"Splits : Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Préparation pipeline
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features)]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    
    # MLflow run
    with mlflow.start_run(run_name="RF_n100"):
        # Paramètres
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", None)
        mlflow.log_param("random_state", config.RANDOM_STATE)
        
        # Entraînement
        logger.info("Entraînement modèle")
        model.fit(X_train, y_train)
        
        # Prédictions et métriques
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        metrics = {
            'accuracy_train': accuracy_score(y_train, y_train_pred),
            'accuracy_val': accuracy_score(y_val, y_val_pred),
            'f1_train': f1_score(y_train, y_train_pred),
            'f1_val': f1_score(y_val, y_val_pred),
        }
        
        # Log métriques MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        metrics['hyperparameters'] = {
            'model_type': 'RandomForest',
            'n_estimators': 100,
            'random_state': config.RANDOM_STATE,
            'test_size': config.TEST_SPLIT,
            'val_size': config.VAL_SPLIT
        }
        with open(config.METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Accuracy validation : {metrics['accuracy_val']:.3f}")
        
        # Matrice de confusion
        cm = confusion_matrix(y_val, y_val_pred)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Matrice confusion validation")
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")
        
        # Sauvegarde modèle MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Artefacts locaux pour API
        Path(config.ARTIFACTS_PATH).mkdir(exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)
        
        # Sauvegarde métriques
        with open(config.METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Schema features
        feature_schema = {name: str(X.dtypes[name]) for name in feature_names}
        with open(config.FEATURE_SCHEMA_PATH, 'w') as f:
            json.dump(feature_schema, f, indent=2)
        
        # Info run
        run_info = {
            'dataset': 'heart_disease.csv',
            'shape': list(df.shape),
            'splits': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
            'random_state': config.RANDOM_STATE,
            'mlflow_run_id': mlflow.active_run().info.run_id
        }
        with open(config.RUN_INFO_PATH, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        logger.info(f"Run terminé : {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()
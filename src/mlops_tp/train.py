"""Entrainement Heart Disease avec MLflow"""
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # MLflow optionnel selon l'environnement
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("heart_disease_mlops")

    logger.info("Chargement des donnees")
    df = pd.read_csv(config.DATA_PATH)
    logger.info(f"Dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()

    # Split train/val/test avec stratification pour conserver la proportion des classes
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SPLIT, random_state=config.RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VAL_SPLIT/(1-config.TEST_SPLIT),
        random_state=config.RANDOM_STATE, stratify=y_temp
    )

    logger.info(f"Splits : Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Pipeline de preprocessing : imputation des valeurs manquantes + normalisation
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features)]
    )

    # Pipeline complet : preprocessing + modele avec equilibrage des classes
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])

    with mlflow.start_run(run_name="RF_n100_balanced"):

        # Log des parametres
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", None)
        mlflow.log_param("random_state", config.RANDOM_STATE)
        mlflow.log_param("class_weight", "balanced")

        # Entrainement
        logger.info("Entrainement modele")
        model.fit(X_train, y_train)

        # Predictions sur train et validation
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Calcul des metriques
        metrics = {
            'accuracy_train': accuracy_score(y_train, y_train_pred),
            'accuracy_val': accuracy_score(y_val, y_val_pred),
            'f1_train': f1_score(y_train, y_train_pred),
            'f1_val': f1_score(y_val, y_val_pred),
        }

        # Log des metriques dans MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))

        # Ajout du timestamp et des hyperparametres pour la tracabilite
        metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        metrics['hyperparameters'] = {
            'model_type': 'RandomForest',
            'n_estimators': 100,
            'random_state': config.RANDOM_STATE,
            'class_weight': 'balanced',
            'test_size': config.TEST_SPLIT,
            'val_size': config.VAL_SPLIT
        }

        logger.info(f"Accuracy validation : {metrics['accuracy_val']:.3f}")

        # Matrice de confusion sur la validation
        cm = confusion_matrix(y_val, y_val_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Matrice confusion validation")
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        # Sauvegarde du modele dans MLflow
        mlflow.sklearn.log_model(model, "model")

        # Sauvegarde des artefacts locaux pour l'API
        Path(config.ARTIFACTS_PATH).mkdir(exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)

        # Sauvegarde des metriques
        with open(config.METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Schema des features attendues par l'API
        feature_schema = {name: str(X.dtypes[name]) for name in feature_names}
        with open(config.FEATURE_SCHEMA_PATH, 'w') as f:
            json.dump(feature_schema, f, indent=2)

        # Informations sur le run pour la tracabilite
        run_info = {
            'dataset': 'heart_disease.csv',
            'shape': list(df.shape),
            'splits': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
            'random_state': config.RANDOM_STATE,
            'mlflow_run_id': mlflow.active_run().info.run_id
        }
        with open(config.RUN_INFO_PATH, 'w') as f:
            json.dump(run_info, f, indent=2)

        logger.info(f"Run termine : {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()
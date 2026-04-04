"""Lance les 3 runs MLflow pour comparaison - TP2 Q22-Q28"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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
import matplotlib.pyplot as plt
import seaborn as sns
import mlops_tp.config as config
import time

# Configuration des 3 runs a comparer
RUNS = [
    {
        "run_name":     "RF_n100",
        "n_estimators": 100,
        "max_depth":    None,
    },
    {
        "run_name":     "RF_n200",
        "n_estimators": 200,
        "max_depth":    None,
    },
    {
        "run_name":     "RF_depth5",
        "n_estimators": 100,
        "max_depth":    5,
    },
]

def run_experiment(run_name, n_estimators, max_depth):
    """Execute un run MLflow avec les parametres donnes"""

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("heart_disease_mlops")

    # Chargement des donnees
    df = pd.read_csv(config.DATA_PATH)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=config.VAL_SPLIT / (1 - config.TEST_SPLIT),
        random_state=config.RANDOM_STATE,
        stratify=y_temp
    )

    # Pipeline preprocessing + modele
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features)]
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier',   RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    with mlflow.start_run(run_name=run_name):

        # Log des parametres
        mlflow.log_param("model_type",    "RandomForest")
        mlflow.log_param("n_estimators",  n_estimators)
        mlflow.log_param("max_depth",     max_depth)
        mlflow.log_param("random_state",  config.RANDOM_STATE)

        # Entrainement
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred   = model.predict(X_val)

        # Calcul des metriques
        metrics = {
            'accuracy_train': accuracy_score(y_train, y_train_pred),
            'accuracy_val':   accuracy_score(y_val,   y_val_pred),
            'f1_train':       f1_score(y_train, y_train_pred),
            'f1_val':         f1_score(y_val,   y_val_pred),
        }

        # Log des metriques MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))

        # Matrice de confusion
        cm = confusion_matrix(y_val, y_val_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Matrice de confusion - {run_name}")
        cm_path = f"confusion_matrix_{run_name}.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # Log du modele
        mlflow.sklearn.log_model(model, "model")

        print(f"  Run : {run_name}")
        print(f"  accuracy_train : {metrics['accuracy_train']:.3f}")
        print(f"  accuracy_val   : {metrics['accuracy_val']:.3f}")
        print(f"  f1_train       : {metrics['f1_train']:.3f}")
        print(f"  f1_val         : {metrics['f1_val']:.3f}")
        print(f"  run_id         : {mlflow.active_run().info.run_id}")
        print()


if __name__ == "__main__":
    print("Lancement des 3 runs MLflow\n")
    for run_config in RUNS:
        print(f"--- {run_config['run_name']} ---")
        run_experiment(
            run_name=     run_config["run_name"],
            n_estimators= run_config["n_estimators"],
            max_depth=    run_config["max_depth"],
        )
    print("Tous les runs sont termines.")
    print("Consulte les resultats sur http://localhost:5000")
"""Configuration du projet MLOps"""
import os

# Chemins ABSOLUS depuis la racine TP_MLOps/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_disease.csv")
ARTIFACTS_PATH = os.path.join(BASE_DIR, "src", "mlops_tp", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_PATH, "model.joblib")
METRICS_PATH = os.path.join(ARTIFACTS_PATH, "metrics.json")
FEATURE_SCHEMA_PATH = os.path.join(ARTIFACTS_PATH, "feature_schema.json")
RUN_INFO_PATH = os.path.join(ARTIFACTS_PATH, "run_info.json")

# Hyperparamètres
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42
MODEL_VERSION = "0.1.0"

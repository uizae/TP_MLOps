# TP MLOps - Détection Maladie Cardiaque

## Dataset Heart Disease UCI
Dataset médical pour prédire maladie cardiaque.  
303 lignes, 14 colonnes.  
**Tâche** : classification (0=pas malade, 1=malade).  
**Source** : UCI Machine Learning Repository (open source)  
[https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

## Installation
```powershell
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt

# Entraîner modèle
python src/mlops_tp/train.py

# API
uvicorn src.mlops_tp.api:app --host 127.0.0.1 --port 8000 --reload

# Tests
pytest tests/
```

## Performances
**Meilleur modèle** : RandomForest n_estimators=100  
- Accuracy validation : 78.3%  
- F1-score validation : 80.8%

## MLflow - 3 tests (TP2 questions 22-28)

**Q22 - Ce qui change dans chaque test :**
| Test | n_estimators | max_depth | accuracy_val | f1_val |
|------|--------------|-----------|--------------|--------|
| Run 1 | 100 | None | **0.783** | **0.808** |
| Run 2 | 200 | None | 0.761 | 0.784 |
| Run 3 | 100 | 5 | 0.739 | 0.769 |

**Q23 - Pourquoi ces tests ?**  
- Test 1 : configuration de base  
- Test 2 : plus d'arbres (200) pour voir si améliore  
- Test 3 : profondeur limitée (5) pour éviter sur-apprentissage  

**Q24-25 - Meilleur test ?**  
**Run 1** : accuracy_val=0.783 et f1_val=0.808 (les 2 meilleures)

**Q26 - Problèmes observés :**  
- Run 2 (200 arbres) : accuracy plus basse  
- Run 3 (depth=5) : très bon sur entraînement (95%) mais mauvais sur validation (73%) → sur-apprentissage

**Q27 - Une seule métrique suffit ?**  
Non. Accuracy seule ne voit pas déséquilibre classes. F1-score plus complet.

**Q28 - Configuration choisie :**  
**n_estimators=100, max_depth=None** (Run 1)

## MLflow UI
```powershell
mlflow ui --host 0.0.0.0 --port 5000
```
http://localhost:5000 → experiment `heart_disease_mlops`

## MLflow Docker
```powershell
docker-compose up -d

## Tests pytest 
```powershell
docker-compose up -d
pytest tests/ -v
```
**5/5 passed** : API, modèle, métriques
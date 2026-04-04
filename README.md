# TP MLOps - Détection de maladie cardiaque

## Dataset Heart Disease UCI

Ce dataset médical contient des informations sur des patients et permet de prédire
si une personne a une maladie cardiaque ou non.

- 303 lignes et 14 colonnes
- **Tâche** : classification binaire (0 = pas malade, 1 = malade)
- **Licence** : open source, UCI Machine Learning Repository
- **Source** : [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

Les 13 variables d'entrée décrivent des caractéristiques médicales comme l'âge,
le sexe, la fréquence cardiaque maximale ou le taux de cholestérol.
La variable cible est `target` (0 ou 1).

Les principaux défis de ce dataset sont sa petite taille (303 lignes),
un léger déséquilibre entre les classes, et quelques valeurs manquantes
traitées par imputation médiane dans le pipeline.

## Installation
```powershell
# Créer et activer l'environnement virtuel
python -m venv .venv
.\.venv\Scripts\activate.bat

# Installer les dépendances
pip install -r requirements.txt

# Entraîner le modèle (génère les artefacts dans src/mlops_tp/artifacts/)
python src/mlops_tp/train.py

# Lancer l'API
uvicorn src.mlops_tp.api:app --host 127.0.0.1 --port 8000 --reload

# Lancer les tests
pytest tests/ -v
```

## Performances du modèle

Le meilleur modèle obtenu est un RandomForest avec 100 arbres de décision.

- Accuracy sur la validation : **78.3%**
- F1-score sur la validation : **80.8%**

Ces résultats sont corrects pour un dataset de cette taille.
Le F1-score est la métrique principale car les erreurs de type
"faux négatif" (patient malade non détecté) sont plus critiques
dans un contexte médical.

## MLflow - Comparaison des expérimentations (TP2 Q22-Q28)

MLflow permet de suivre et comparer plusieurs entraînements de manière organisée.
Pour lancer les 3 runs d'un seul coup :
```powershell
# Démarrer le serveur MLflow dans un terminal séparé
mlflow ui --host 0.0.0.0 --port 5000

# Lancer les 3 expérimentations dans un autre terminal
python run_experiments.py
```

Résultats disponibles sur : http://localhost:5000 → expérience `heart_disease_mlops`

**Q22 - Résultats des 3 runs :**

| Run | n_estimators | max_depth | accuracy_train | accuracy_val | f1_train | f1_val |
|-----|-------------|-----------|---------------|--------------|----------|--------|
| RF_n100 | 100 | None | 1.000 | **0.783** | 1.000 | **0.808** |
| RF_n200 | 200 | None | 1.000 | 0.761 | 1.000 | 0.784 |
| RF_depth5 | 100 | 5 | 0.957 | 0.739 | 0.961 | 0.769 |

**Q23 - Pourquoi ces variations ?**

Le Run 1 sert de référence avec une configuration de base.
Le Run 2 teste si augmenter le nombre d'arbres à 200 améliore les résultats.
Le Run 3 teste si limiter la profondeur des arbres à 5 réduit le sur-apprentissage.

**Q24-Q25 - Meilleur run et métrique choisie :**

Le Run 1 (RF_n100) est le meilleur avec accuracy_val=0.783 et f1_val=0.808.
Le choix se base principalement sur le f1_val car dans un contexte médical,
il est important de bien détecter les vrais positifs et de limiter les faux négatifs.

**Q26 - Compromis observés :**

Le Run 2 avec 200 arbres n'améliore pas les performances sur la validation
malgré un temps d'entraînement plus long. Le Run 3 avec max_depth=5 réduit
légèrement le sur-apprentissage (accuracy_train passe de 1.0 à 0.957) mais
dégrade les performances sur la validation (0.739 contre 0.783).

**Q27 - Une seule métrique suffit-elle ?**

Non, une seule métrique ne suffit pas. Si on regarde uniquement accuracy_train,
les runs 1 et 2 semblent identiques (1.0 tous les deux). C'est en comparant
train et validation ensemble qu'on détecte le sur-apprentissage. De plus,
l'accuracy seule peut être trompeuse si les classes sont déséquilibrées,
c'est pourquoi on utilise aussi le F1-score.

**Q28 - Configuration retenue :**

On retient le Run 1 : n_estimators=100, max_depth=None, random_state=42.
C'est la configuration qui donne les meilleures performances sur la validation
avec le meilleur équilibre entre accuracy et F1-score.

## Lancer avec Docker
```powershell
# Démarrer tous les services (API + Frontend + MLflow)
docker-compose up -d

# Vérifier que les conteneurs tournent
docker ps
```

Services disponibles :
- API FastAPI : http://localhost:8000
- Interface Streamlit : http://localhost:8501
- MLflow UI : http://localhost:5000

## Tests
```powershell
pytest tests/ -v
```

**8 tests passent** : santé de l'API, prédictions, erreurs 422, inférence,
chargement du modèle et vérification des métriques.
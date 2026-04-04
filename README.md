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

J'ai choisi ce dataset car le domaine médical me semblait intéressant pour apprendre
le MLOps : les erreurs de prédiction ont des conséquences réelles, ce qui oblige à
bien réfléchir au choix des métriques et à la qualité du modèle.

Les principaux défis sont la petite taille du dataset (303 lignes seulement),
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

---

## TP2 - Suivi des expérimentations avec MLflow

### Qu'est-ce que MLflow ?

C'est la première fois que j'utilise un outil de suivi d'expérimentations.
MLflow permet d'enregistrer automatiquement ce qu'on a testé, les résultats obtenus
et les fichiers produits. Avant de l'utiliser, je notais les résultats à la main
dans un fichier texte, ce qui était peu fiable et difficile à comparer.

### Questions préliminaires

**Q1 - Qu'est-ce qu'une expérience dans MLflow ?**

Une expérience est un regroupement logique de plusieurs runs. Elle correspond
à un projet ou un objectif précis. Dans mon projet, l'expérience s'appelle
`heart_disease_mlops` et regroupe tous mes essais d'entraînement du modèle.
C'est un peu comme un dossier qui contient toutes mes tentatives.

**Q2 - Qu'est-ce qu'un run ?**

Un run est une exécution unique du script d'entraînement. Chaque fois que je lance
`train.py`, un nouveau run est créé dans l'expérience avec un identifiant unique.
Il enregistre automatiquement les paramètres, les métriques et les artefacts
de cette exécution. Ce que j'ai trouvé pratique, c'est que tout est horodaté
et qu'on peut retrouver n'importe quel run passé.

**Q3 - Différence entre paramètre, métrique et artefact ?**

Un paramètre est une valeur qu'on fixe avant l'entraînement pour contrôler
le comportement du modèle, par exemple `n_estimators=100`. On le choisit avant
de lancer l'entraînement.

Une métrique est une valeur calculée après l'entraînement pour évaluer les
performances, comme `accuracy_val=0.783`. Elle mesure si le modèle apprend bien.

Un artefact est un fichier produit pendant ou après l'entraînement, comme
`confusion_matrix.png` ou `model.joblib`. C'est un fichier qu'on veut garder
et pouvoir réutiliser.

**Q4 - Exemples dans mon projet :**

Trois paramètres que j'enregistre : `n_estimators` (nombre d'arbres),
`max_depth` (profondeur maximale des arbres), `random_state` (graine aléatoire
pour la reproductibilité).

Deux métriques pertinentes : `accuracy_val` (taux de bonnes prédictions sur
la validation) et `f1_val` (F1-score sur la validation, plus adapté au déséquilibre
de classes).

Un artefact utile : `confusion_matrix.png` qui montre visuellement les types
d'erreurs commises par le modèle.

**Q5 - Sur quelle adresse l'interface MLflow est-elle accessible ?**

L'interface MLflow est accessible à `http://localhost:5000` quand on la lance
en local. Dans Docker, le service MLflow expose ce même port via le
`docker-compose.yml`. C'est une interface web qui permet de visualiser
tous les runs et de les comparer.

**Q6 - Que remarque-t-on avant le premier run ?**

L'interface est complètement vide. Aucun run n'apparaît et l'expérience
`heart_disease_mlops` n'existe pas encore. Seule l'expérience par défaut
`Default` est présente. J'ai trouvé ça logique : MLflow crée l'expérience
uniquement quand on lance le premier entraînement.

### Lancer les expérimentations
```powershell
# Démarrer le serveur MLflow dans un terminal séparé
mlflow ui --host 0.0.0.0 --port 5000

# Lancer les 3 expérimentations dans un autre terminal
python run_experiments.py
```

Résultats disponibles sur : http://localhost:5000 -> expérience `heart_disease_mlops`

### Intégration MLflow dans le code

**Q13 - Quels paramètres ai-je choisi d'enregistrer ?**

J'enregistre `model_type`, `n_estimators`, `max_depth` et `random_state`.

**Q14 - Pourquoi ces paramètres ?**

Ce sont les paramètres qui influencent le plus les performances et la
reproductibilité. `n_estimators` contrôle la complexité du modèle et le temps
d'entraînement. `max_depth` contrôle le risque de sur-apprentissage.
`random_state` garantit qu'on peut reproduire exactement le même résultat,
ce qui est important pour comparer les runs de manière fiable.

**Q15 - Quelles métriques ai-je retenues ?**

J'enregistre `accuracy_train`, `accuracy_val`, `f1_train` et `f1_val`.

**Q16 - Pourquoi ces métriques ?**

L'accuracy mesure le taux global de bonnes prédictions mais peut être trompeuse
si les classes sont déséquilibrées. Le F1-score est plus fiable car il tient
compte à la fois de la précision et du rappel. Avoir les métriques sur le train
ET la validation est important pour détecter le sur-apprentissage : si accuracy_train
est très élevée mais accuracy_val est basse, c'est un signe que le modèle a mémorisé
les données au lieu d'apprendre.

### Artefact enregistré

**Q18 - Quel artefact ai-je choisi ?**

La matrice de confusion (`confusion_matrix.png`) générée sur le jeu de validation.

**Q19 - Pourquoi cet artefact ?**

Elle montre non seulement le nombre de bonnes prédictions, mais aussi le type
d'erreurs commises. Dans un contexte médical, les faux négatifs (patient malade
déclaré sain) sont particulièrement critiques car ils peuvent avoir des conséquences
graves. La matrice de confusion permet de voir directement combien de ces erreurs
critiques le modèle commet.

**Q20 - À quel moment est-il produit ?**

Il est produit juste après l'évaluation du modèle sur le jeu de validation,
avant la sauvegarde des artefacts locaux.

**Q21 - Vérification dans l'interface MLflow :**

Après exécution de `train.py`, j'ai vérifié dans l'UI que le run `RF_n100`
contient bien les 4 paramètres dans l'onglet "Parameters", les 4 métriques
dans l'onglet "Metrics", la `confusion_matrix.png` dans l'onglet "Artifacts",
et le dossier `model/` avec le modèle sklearn sérialisé.

### Comparaison de 3 runs

**Q22 - Résultats des 3 runs :**

| Run | n_estimators | max_depth | accuracy_train | accuracy_val | f1_train | f1_val |
|-----|-------------|-----------|---------------|--------------|----------|--------|
| RF_n100 | 100 | None | 1.000 | **0.783** | 1.000 | **0.808** |
| RF_n200 | 200 | None | 1.000 | 0.761 | 1.000 | 0.784 |
| RF_depth5 | 100 | 5 | 0.957 | 0.739 | 0.961 | 0.769 |

**Q23 - Pourquoi ces variations ?**

Le Run 1 sert de référence avec une configuration de base. Le Run 2 teste si
augmenter le nombre d'arbres à 200 améliore les résultats : j'imaginais
qu'avoir plus d'arbres donnerait un modèle plus précis. Le Run 3 teste si
limiter la profondeur des arbres à 5 réduit le sur-apprentissage, car des arbres
trop profonds peuvent mémoriser les données d'entraînement.

**Q24-Q25 - Meilleur run et métrique choisie :**

Le Run 1 (RF_n100) est le meilleur avec accuracy_val=0.783 et f1_val=0.808.
Le choix se base principalement sur le f1_val car dans un contexte médical,
il est important de bien détecter les vrais positifs et de limiter les faux négatifs.

**Q26 - Compromis observés :**

Le Run 2 avec 200 arbres n'améliore pas les performances sur la validation malgré
un temps d'entraînement plus long. C'était surprenant : j'aurais pensé que plus
d'arbres = meilleurs résultats. Le Run 3 avec max_depth=5 réduit légèrement le
sur-apprentissage (accuracy_train passe de 1.0 à 0.957) mais dégrade les
performances sur la validation (0.739 contre 0.783).

**Q27 - Une seule métrique suffit-elle ?**

Non, une seule métrique ne suffit pas. Si on regarde uniquement accuracy_train,
les runs 1 et 2 semblent identiques (1.0 tous les deux). C'est en comparant
train et validation ensemble qu'on détecte le sur-apprentissage. De plus,
l'accuracy seule peut être trompeuse si les classes sont déséquilibrées,
c'est pourquoi j'utilise aussi le F1-score.

**Q28 - Configuration retenue :**

On retient le Run 1 : n_estimators=100, max_depth=None, random_state=42.
C'est la configuration qui donne les meilleures performances sur la validation
avec le meilleur équilibre entre accuracy et F1-score.

---

## TP3 - Déploiement et CI/CD

### Questions préliminaires

**Q1 - Différence entre tester localement et déployer ?**

Tester localement signifie faire tourner l'application sur sa propre machine,
avec ses propres fichiers et sa propre configuration. Déployer signifie mettre
l'application sur un serveur distant accessible par n'importe qui via une URL
publique. Ce que j'ai appris dans ce TP, c'est qu'une application qui fonctionne
en local peut très bien échouer une fois déployée si elle dépend de chemins
locaux ou de fichiers présents uniquement sur notre machine.

**Q2 - Rôle du Dockerfile dans une chaîne CI/CD ?**

Le Dockerfile décrit comment construire l'image de l'application de manière
reproductible. Dans une chaîne CI/CD, il garantit que l'application s'exécute
dans le même environnement à chaque déploiement, peu importe la machine qui
fait le build. Sans Docker, on aurait des différences entre les environnements
qui rendraient les déploiements imprévisibles.

**Q3 - Pourquoi une app qui fonctionne en local peut échouer une fois déployée ?**

Parce qu'elle peut dépendre de chemins locaux codés en dur, de fichiers présents
uniquement sur la machine du développeur, d'un port fixe, ou de variables
d'environnement non définies sur le serveur distant. Dans mon projet, j'avais
ce problème avec le chemin vers `model.joblib` et l'adresse du serveur MLflow.

**Q4 - Rôle du endpoint /health ?**

Il permet de vérifier que le service est vivant et prêt à répondre. Sur Render,
ce endpoint est utilisé pour les health checks automatiques : si `/health` ne
répond pas correctement, la plateforme considère que le service est en panne
et peut tenter de le redémarrer. C'est aussi utile pour vérifier rapidement
qu'un déploiement s'est bien passé.

**Q5 - Différence entre CI et CD ?**

La CI (Intégration Continue) automatise les vérifications à chaque modification
du code : tests, build Docker, analyse de qualité. La CD (Déploiement Continu)
automatise la mise en production du code validé par la CI. En résumé, la CI
vérifie que le code est correct, la CD le met en ligne automatiquement si c'est
le cas.

### Partie 1 - Déploiement sur Render

L'API est déployée et accessible publiquement à cette adresse :
**https://heart-disease-api-n9jg.onrender.com**

Routes disponibles :
- `/health` : vérifie que le service est en ligne
- `/predict` : envoie des données patient et reçoit une prédiction
- `/metadata` : informations sur le modèle
- `/docs` : interface Swagger pour tester l'API

Le modèle est entraîné automatiquement au moment du build Docker grâce à la
commande `RUN python src/mlops_tp/train.py` dans le Dockerfile.

### Partie 2 - CI avec GitHub Actions

**Q22 - Que signifie un pipeline vert ? Rouge ?**

Un pipeline vert signifie que toutes les étapes ont réussi : les tests passent
et l'image Docker se build correctement. On peut déployer en confiance.
Un pipeline rouge signifie qu'une étape a échoué. Il faut lire les logs pour
identifier et corriger l'erreur avant de pouvoir déployer. J'ai volontairement
provoqué un échec pendant le TP pour voir à quoi ça ressemble, c'est très
visible dans l'onglet Actions de GitHub.

**Q23 - Pourquoi exécuter les tests avant le déploiement ?**

Pour éviter de mettre en production un code cassé. Si les tests échouent,
le déploiement ne se fait pas. Cela protège les utilisateurs et garantit
la qualité du service. C'est une des leçons principales de ce TP : on ne
déploie que du code vérifié.

**Q24 - Pourquoi builder l'image Docker dans la CI ?**

Pour vérifier que le Dockerfile fonctionne correctement dans un environnement
propre et reproductible, et pas seulement sur ma machine. Si le build échoue
en CI, on le sait avant le déploiement. Cela évite les mauvaises surprises
au moment de mettre en production.

**Q25 - Différence entre push et pull_request comme déclencheurs ?**

Un `push` sur `main` déclenche la CI quand du code est directement envoyé sur
la branche principale. Une `pull_request` déclenche la CI avant que le code
soit fusionné, ce qui permet de vérifier qu'il ne casse rien avant de l'intégrer.
Dans un vrai projet en équipe, on utilise surtout les pull_requests pour que
personne ne puisse casser le code de production sans que la CI valide d'abord.

**Q26 - Première étape qu'on regarde quand un workflow échoue ?**

On regarde l'étape marquée en rouge dans l'onglet Actions de GitHub. On clique
dessus pour lire les logs détaillés et identifier le message d'erreur exact.
En général, l'erreur est assez claire : un test qui échoue, un module manquant,
ou un problème de syntaxe dans le Dockerfile.

Le workflow CI utilisé :
- Déclencheurs : `push` et `pull_request` sur `main`
- Tests lancés : `test_training.py` et `test_inference.py`
- `test_api.py` n'est pas lancé en CI car il nécessite un serveur qui tourne
- Le build Docker ne se lance que si les tests passent (`needs: tests`)

### Partie 3 - CD

**Q28 - Mécanisme de redéploiement automatique :**

Render est connecté au dépôt GitHub. Dès qu'un push est fait sur `main`,
Render détecte le changement et lance automatiquement un nouveau déploiement
à partir du Dockerfile. C'est l'approche simple décrite dans le TP.

**Q29-Q32 - Test du redéploiement :**

J'ai modifié le endpoint `/health` en ajoutant `"version": "1.0.1"`.
Après le push sur `main`, la CI GitHub Actions s'est exécutée en vert,
puis Render a automatiquement redéployé le service. La nouvelle version
était visible sur l'URL publique quelques minutes après le push.

**Q33 - Continuous Deployment vs Continuous Delivery ?**

On parle de Continuous Deployment quand chaque push validé par la CI est
automatiquement mis en production sans intervention humaine, comme dans notre
projet avec Render. On parle de Continuous Delivery quand le code est prêt
à être déployé à tout moment mais qu'un humain doit valider manuellement
le déploiement final. Notre projet utilise le Continuous Deployment.

### Partie 4 - Variables d'environnement et secrets

**Q34 - Variables d'environnement utilisées :**

`MLFLOW_TRACKING_URI` pour l'adresse du serveur MLflow, `MODEL_VERSION` pour
la version du modèle, `PYTHONPATH` pour le chemin des modules Python.

**Q35 - Variables de configuration fonctionnelle :**

`MLFLOW_TRACKING_URI`, `MODEL_VERSION` et `PYTHONPATH`. Ces valeurs changent
selon l'environnement (local, Docker, Render) mais ne sont pas sensibles.
On peut les partager sans risque.

**Q36 - Variables qui relèvent d'un secret :**

Dans ce projet, il n'y a pas encore de secrets car il n'y a pas de base de
données ni de clé API externe. Si on ajoutait une authentification à l'API
ou une connexion à un registre Docker privé, les tokens et mots de passe
deviendraient des secrets à gérer via les variables d'environnement de Render,
jamais dans le code.

**Q37 - Pourquoi ne jamais versionner un secret dans Git ?**

Parce que Git conserve l'historique de tous les commits. Même si on supprime
le secret dans un commit suivant, il reste visible dans l'historique.
N'importe qui ayant accès au dépôt peut le retrouver. C'est une erreur
classique que j'ai découverte dans ce TP et que je ferai attention à éviter.

**Q38 - Que doit contenir un fichier .env local ?**

Le fichier `.env` local contient les variables de configuration pour le
développement, comme `MLFLOW_TRACKING_URI=http://localhost:5000`. Il ne doit
jamais être commité dans un dépôt public car il peut contenir des informations
sensibles. C'est pourquoi il est dans le `.gitignore`. Sur Render, ces variables
sont définies directement dans l'interface de la plateforme.

---

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

**9 tests passent** : santé de l'API, prédictions, erreurs 422, inférence,
chargement du modèle et vérification des métriques.
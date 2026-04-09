# Loan MLOps Project

Ce projet met en œuvre un système automatisé de prédiction du risque de crédit (défaut de prêt). Il suit les principes MLOps pour assurer la traçabilité, la reproductibilité et la qualité des modèles.

## Architecture du Projet
Le pipeline est divisé en trois étapes clés :

1.  **Preprocessing (`01_preprocessing.py`)** :
    *   Nettoyage et ingénierie des features (ratios financiers, interactions).
    *   Génération de deux jeux de données pour tester différentes profondeurs d'analyse :
        *   `full_features` : Scénario avec un volume anormalement élevé de données client (analyse exhaustive).
        *   `safe_features` : Scénario avec le contexte d'information client habituel (données standard).
    *   Division en jeux d'entraînement, validation et test.

2.  **Entraînement (`02_training.py`)** :
    *   Entraînement de modèles (Logistic Regression, Random Forest, XGBoost).
    *   Optimisation hyperparamétrique (Grid/Randomized Search).
    *   Optimisation du seuil de décision pour maximiser le **score F2** (priorité au rappel).
    *   Tracking automatique via **MLflow**.

3.  **Évaluation (`03_evaluation.py`)** :
    *   Comparaison automatique des meilleures versions de chaque modèle pour un mode de données choisi.
    *   Évaluation sur le jeu de test final.

## Dashboard (`app.py`)
Application **Streamlit** pour :
*   Explorer les données (visualisations plotly).
*   Suivre les performances des modèles via MLflow.

## Démarrage rapide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Pipeline
*   **Nettoyage** : `python 01_preprocessing.py`
*   **Entraînement** : `python 02_training.py`
*   **Évaluation** : `python 03_evaluation.py`

### 3. Interface
```bash
# Dashboard Streamlit
streamlit run app.py

# Interface MLflow
mlflow ui
```

import pandas as pd 
import numpy as np
import mlflow 
import mlflow.sklearn
import tempfile
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, fbeta_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
DATASET_OPTIONS = {
    "1": "full_features",
    "2": "safe_features",
}


def choose_dataset_mode():
    print("Select training dataset:")
    print("1. full_features")
    print("2. safe_features")
    choice = input("Choice [1/2] (default: 1): ").strip() or "1"
    if choice in DATASET_OPTIONS:
        return DATASET_OPTIONS[choice]
    if choice in DATASET_OPTIONS.values():
        return choice
    raise ValueError("Invalid dataset choice. Use 1, 2, full_features, or safe_features.")


def load_split(split_dir):
    X_train = pd.read_csv(f"{split_dir}/X_train.csv")
    X_val = pd.read_csv(f"{split_dir}/X_val.csv")
    y_train = pd.read_csv(f"{split_dir}/y_train.csv").values.ravel()
    y_val = pd.read_csv(f"{split_dir}/y_val.csv").values.ravel()
    return X_train, X_val, y_train, y_val

param_grids = {
    "Logistic_Regression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse de la force de régularisation (échelle log)
        "solver": ["liblinear"]              # Algorithme adapté aux petits datasets/régularisation
    },
    "Random_Forest": {
        "n_estimators": [100, 200, 300],     # Nombre d'arbres (compromis performance/temps)
        "max_depth": [5, 10, 20, None],      # Profondeur max des arbres (None = expansion totale)
        "min_samples_split": [2, 5, 10],     # Nb min d'échantillons pour diviser un nœud
        "min_samples_leaf": [1, 2, 4]        # Nb min d'échantillons requis dans une feuille
    }, 
    "XGBoost": {
        "n_estimators": [100, 200, 300],     # Nombre d'arbres de boosting
        "learning_rate": [0.03, 0.05, 0.1, 0.2], # Pas d'apprentissage (vitesse de convergence)
        "max_depth": [3, 4, 5, 6],           # Profondeur des arbres (contrôle le sur-apprentissage)
        "subsample": [0.8, 1.0],             # Proportion d'échantillons par arbre (robustesse)
        "colsample_bytree": [0.8, 1.0]       # Proportion de colonnes par arbre (diversité)
    }
}

models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random_Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "XGBoost": XGBClassifier(random_state=42, scale_pos_weight=4.2)
}

model_pip_requirements = [
    "mlflow==3.10.1",
    "scikit-learn==1.8.0",
    "skops==0.13.0",
    "xgboost==3.2.0"
]

trusted_xgboost_types = [
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier"
]

search_iterations = {
    "Random_Forest": 20,
    "XGBoost": 30
}

scoring_metrics = {
    "f2": make_scorer(fbeta_score, beta=2),
    "f1": make_scorer(f1_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "accuracy": make_scorer(accuracy_score)
}

# Stratified CV preserves the default/non-default ratio in each fold.
cv_strategy = StratifiedKFold(n_splits=5, shuffle=False)

dataset_mode = choose_dataset_mode()
split_dir = f"data/processed/{dataset_mode}"
X_train, X_val, y_train, y_val = load_split(split_dir)

print(f"Training with dataset mode: {dataset_mode}")

for model_name, model_obj in models.items(): 
    mlflow.set_experiment(f"Loan_Default_{model_name}")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{model_name}_{dataset_mode}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("feature_mode", dataset_mode)
        total_candidates = len(list(ParameterGrid(param_grids[model_name])))

        if model_name == "Logistic_Regression":
            cv_search = GridSearchCV(
                estimator=model_obj,
                param_grid=param_grids[model_name],
                cv=cv_strategy,
                scoring=scoring_metrics,
                refit="f2",
                n_jobs=-1
            )
        else:
            n_iter = min(search_iterations[model_name], total_candidates)
            cv_search = RandomizedSearchCV(
                estimator=model_obj,
                param_distributions=param_grids[model_name],
                n_iter=n_iter,
                cv=cv_strategy,
                scoring=scoring_metrics,
                refit="f2",
                n_jobs=-1,
                random_state=42
            )

        cv_search.fit(X_train, y_train)
        best_model = cv_search.best_estimator_

        cv_results = (
            pd.DataFrame(cv_search.cv_results_)
            .sort_values("rank_test_f2")
            .reset_index(drop=True)
        )

        # 2. Find best threshold for F2
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 81)
        
        best_val_f2 = 0 # Initialize this!
        final_thresh = 0.5
        
        for t in thresholds:
            current_f2 = fbeta_score(y_val, (y_val_proba >= t).astype(int), beta=2)
            if current_f2 > best_val_f2:
                best_val_f2 = current_f2
                final_thresh = t

        # 3. Log everything to MLflow
        y_val_pred = (y_val_proba >= final_thresh).astype(int)
        
        mlflow.log_params(cv_search.best_params_)
        mlflow.log_param("optimized_threshold", round(final_thresh, 3))
        mlflow.log_param("target_metric", "F2_Score")
        mlflow.log_metric("cv_best_f2", cv_search.best_score_)
        mlflow.log_metric("cv_best_f1", cv_results.loc[0, "mean_test_f1"])
        mlflow.log_metric("cv_best_precision", cv_results.loc[0, "mean_test_precision"])
        mlflow.log_metric("cv_best_recall", cv_results.loc[0, "mean_test_recall"])
        mlflow.log_metric("cv_best_accuracy", cv_results.loc[0, "mean_test_accuracy"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            cv_results_path = Path(tmp_dir) / f"{model_name}_cv_results.csv"
            cv_results.to_csv(cv_results_path, index=False)
            mlflow.log_artifact(str(cv_results_path), artifact_path="cv_results")

        # Log both metrics so you can see the trade-off in the UI
        mlflow.log_metric("val_f2", best_val_f2)
        mlflow.log_metric("val_f1", f1_score(y_val, y_val_pred))
        mlflow.log_metric("val_recall", recall_score(y_val, y_val_pred))
        mlflow.log_metric("val_precision", precision_score(y_val, y_val_pred))
        mlflow.log_metric("val_accuracy", accuracy_score(y_val, y_val_pred))
        mlflow.log_metric("val_auc", roc_auc_score(y_val, y_val_proba))

        log_model_kwargs = {
            "sk_model": best_model,
            "name": "model",
            "serialization_format": "skops",
            "pip_requirements": model_pip_requirements
        }
        if model_name == "XGBoost":
            log_model_kwargs["skops_trusted_types"] = trusted_xgboost_types

        mlflow.sklearn.log_model(**log_model_kwargs)

        print(f"{model_name} Optimized for F2!")
        print(f"   Best Thresh: {final_thresh:.3f} | Val F2: {best_val_f2:.4f}")
        print(f"   (Resulting Val Recall: {recall_score(y_val, y_val_pred):.4f})")

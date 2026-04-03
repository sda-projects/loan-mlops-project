import pandas as pd 
import numpy as np
import mlflow 
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score, fbeta_score, accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. Load Data
X_train = pd.read_csv("data/processed/X_train.csv")
X_val = pd.read_csv("data/processed/X_val.csv")    


y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()



param_grids = {
    "Logistic_Regression": {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear"]},
    "Random_Forest": {"n_estimators": [100, 200], "max_depth": [10, 20]}, 
    "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.1, 0.2], "max_depth": [3, 5]}
}

models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random_Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
    "XGBoost": XGBClassifier(random_state=42, scale_pos_weight=4.2)
}
f2_scorer = make_scorer(fbeta_score, beta=2)

for model_name, model_obj in models.items(): 
    mlflow.set_experiment(f"Loan_Default_{model_name}")

    with mlflow.start_run(run_name="F2_Optimization_Phase"):
        
        # 1. Update CV to use F2
        cv_search = RandomizedSearchCV(
            estimator=model_obj, 
            param_distributions=param_grids[model_name],
            n_iter=5, cv=5, 
            scoring=f2_scorer, # <--- Matches your strategy
            n_jobs=-1, random_state=42
        )
        cv_search.fit(X_train, y_train)
        best_model = cv_search.best_estimator_

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

        # Log both metrics so you can see the trade-off in the UI
        mlflow.log_metric("val_f2", best_val_f2)
        mlflow.log_metric("val_f1", f1_score(y_val, y_val_pred))
        mlflow.log_metric("val_recall", recall_score(y_val, y_val_pred))
        mlflow.log_metric("val_precision", precision_score(y_val, y_val_pred))
        mlflow.log_metric("val_auc", roc_auc_score(y_val, y_val_proba))

        mlflow.sklearn.log_model(best_model, "model")

        print(f"✅ {model_name} Optimized for F2!")
        print(f"   Best Thresh: {final_thresh:.3f} | Val F2: {best_val_f2:.4f}")
        print(f"   (Resulting Val Recall: {recall_score(y_val, y_val_pred):.4f})")
import pandas as pd 
import numpy as np
import mlflow 
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. Load Data
X_train = pd.read_csv("data/processed/X_train.csv")
X_val = pd.read_csv("data/processed/X_val.csv")    
X_test = pd.read_csv("data/processed/X_test.csv")  

y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_val = pd.read_csv("data/processed/y_val.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

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

# --- THE TUNING LOOP (VALIDATION ONLY) ---
for model_name, model_obj in models.items(): 
    mlflow.set_experiment(f"Loan_Default_{model_name}")

    with mlflow.start_run(run_name="Val_Based_Tweak"):
        # RandomizedSearch uses its own internal CV on the TRAIN set
        cv_search = RandomizedSearchCV(
            estimator=model_obj, 
            param_distributions=param_grids[model_name],
            n_iter=5, cv=5, scoring='f1', n_jobs=-1, random_state=42
        )
        cv_search.fit(X_train, y_train)
        best_model = cv_search.best_estimator_

        # --- STEP 2: TWEAK THRESHOLD USING ONLY VALIDATION ---
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Scan thresholds to find the one that maximizes VALIDATION F1
        thresholds = np.linspace(0.1, 0.9, 81)
        best_val_f1 = 0
        final_thresh = 0.5
        
        for t in thresholds:
            current_f1 = f1_score(y_val, (y_val_proba >= t).astype(int))
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                final_thresh = t

        # --- STEP 3: LOG VALIDATION RESULTS ---
        y_val_pred = (y_val_proba >= final_thresh).astype(int)
        mlflow.log_params(cv_search.best_params_)
        mlflow.log_param("chosen_threshold", round(final_thresh, 3))
        mlflow.log_metric("val_f1", f1_score(y_val, y_val_pred))
        mlflow.log_metric("val_auc", roc_auc_score(y_val, y_val_proba))

        # --- STEP 4: THE FINAL TEST
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= final_thresh).astype(int)
        
        test_f1 = f1_score(y_test, y_test_pred)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.sklearn.log_model(best_model, "model")

        print(f"✅ {model_name} Optimized!")
        print(f"   Best Thresh (from Val): {final_thresh:.3f}")
        print(f"   Validation F1: {best_val_f1:.4f}")
        print(f"   Final Test F1: {test_f1:.4f}\n")
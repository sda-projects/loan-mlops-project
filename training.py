import pandas as pd 
import mlflow 
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")  

# Load processed data 
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()



# define parameter grids for each model 
param_grids = {
    "Logistic_Regression": {
        "C": [0.01, 0.1, 1, 10], 
        "solver": ["liblinear", "lbfgs"]
    },
    "Random_Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20], 
        "min_samples_split": [2, 5]
    }, 
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }
}

# define the three models 
models = {
    "Logistic_Regression": LogisticRegression(max_iter =1000),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

#Training Loop 
for model_name, model_obj in models.items(): 
    #an experiment per model
    mlflow.set_experiment(f"Loan_Default_{model_name}")

    with mlflow.start_run(run_name="Hyperparameters_Tuning"):
        # set up random search with 5-fold cross-validation
        cv_search = RandomizedSearchCV(
            estimator=model_obj, 
            param_distributions=param_grids[model_name],
            n_iter=5, 
            cv=5, 
            scoring='f1',
            n_jobs=-1, 
            random_state=42
        )

        cv_search.fit(X_train, y_train)
        best_model = cv_search.best_estimator_

        #prediction using the best version found 
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:,1]

    
        #calculate metrics 
        metrics ={
            "accuracy": accuracy_score(y_test, y_pred), 
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred), 
            "auc": roc_auc_score(y_test, y_proba)
        }

        #loc best parameters found during CV 
        mlflow.log_params(cv_search.best_params_)
        mlflow.log_metrics(metrics)

        #log the actual model 
        mlflow.sklearn.log_model(best_model, "model")

        print(f"{model_name} (Tuned) -F1 score: {metrics['f1']:.4f}")
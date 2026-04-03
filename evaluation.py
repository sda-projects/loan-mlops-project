import pandas as pd 
import mlflow.sklearn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, fbeta_score


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Final_Test_Evaluation")

#load test data 
X_test = pd.read_csv("data/processed/X_test.csv")  
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Get the models 
#HERE PASTE THE IDS FOR THE MODELS IN MLFLOW WITH THE BEST THRESHOLD FOUND 
model_registry = {
    "Logistic_Regression": {"run_id": "2dc24f0ed7b54928b3e24d7cf0b08321", "threshold": 0.38},
    "Random_Forest":       {"run_id": "a10b316311a94897a37fa4f2dcb3d9ea", "threshold": 0.21},
    "XGBoost":             {"run_id": "2696a14ded2e4d8284735d70aa4ef712", "threshold": 0.32}
}


final_result = []

print("starting final evaluation for the three models")
      
for name, info in model_registry.items():
    # Start a NEW run specifically for the Test results
    with mlflow.start_run(run_name=f"Test_Eval_{name}"):
        
        # Load the model
        model_uri = f"runs:/{info['run_id']}/model"
        model = mlflow.sklearn.load_model(model_uri)

        # Predict
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= info['threshold']).astype(int)

        # 2. Calculate metrics
        metrics = {
            "test_f1": f1_score(y_test, y_pred),
            "test_f2": fbeta_score(y_test, y_pred, beta=2),
            "test_recall": recall_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_accuracy": accuracy_score(y_test, y_pred)
        }

        # 3. PUSH to MLflow
        mlflow.log_params(info) # Logs the RunID and Threshold used
        mlflow.log_metrics(metrics)
        
        # Keep your local list for a quick printout
        final_result.append({"Model": name, **metrics})

        print(f"✅ {name} Test metrics logged to MLflow!")

# Display the results table at the end
print("\n", pd.DataFrame(final_result))
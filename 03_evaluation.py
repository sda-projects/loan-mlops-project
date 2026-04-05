import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, fbeta_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Final_Test_Evaluation")
DATASET_OPTIONS = {
    "1": "full_features",
    "2": "safe_features",
}


def choose_dataset_mode():
    print("Select evaluation dataset:")
    print("1. full_features")
    print("2. safe_features")
    choice = input("Choice [1/2] (default: 1): ").strip() or "1"
    if choice in DATASET_OPTIONS:
        return DATASET_OPTIONS[choice]
    if choice in DATASET_OPTIONS.values():
        return choice
    raise ValueError("Invalid dataset choice. Use 1, 2, full_features, or safe_features.")


def load_test_split(split_dir):
    X_test = pd.read_csv(f"{split_dir}/X_test.csv")
    y_test = pd.read_csv(f"{split_dir}/y_test.csv").values.ravel()
    return X_test, y_test

training_experiments = {
    "Logistic_Regression": "Loan_Default_Logistic_Regression",
    "Random_Forest": "Loan_Default_Random_Forest",
    "XGBoost": "Loan_Default_XGBoost"
}


def get_latest_finished_run(experiment_name, dataset_mode):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            f"and params.feature_mode = '{dataset_mode}'"
        ),
        order_by=["start_time DESC"],
        max_results=1
    )
    if runs.empty:
        raise ValueError(
            f"No FINISHED runs found for experiment: {experiment_name} "
            f"with feature mode {dataset_mode}"
        )

    run = runs.iloc[0]
    return {
        "run_id": run["run_id"],
        "threshold": float(run["params.optimized_threshold"])
    }


final_result = []
dataset_mode = choose_dataset_mode()
split_dir = f"data/processed/{dataset_mode}"
X_test, y_test = load_test_split(split_dir)

print("starting final evaluation for the three models")
print(f"Evaluation dataset mode: {dataset_mode}")

for name, experiment_name in training_experiments.items():
    info = get_latest_finished_run(experiment_name, dataset_mode)

    with mlflow.start_run(run_name=f"Test_Eval_{name}"):
        mlflow.log_param("feature_mode", dataset_mode)
        model_uri = f"runs:/{info['run_id']}/model"
        model = mlflow.sklearn.load_model(model_uri)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= info["threshold"]).astype(int)

        metrics = {
            "test_f1": f1_score(y_test, y_pred),
            "test_f2": fbeta_score(y_test, y_pred, beta=2),
            "test_recall": recall_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_accuracy": accuracy_score(y_test, y_pred)
        }

        mlflow.log_params(info)
        mlflow.log_metrics(metrics)

        final_result.append({"Model": name, **metrics})
        print(f"{name} Test metrics logged to MLflow!")

print("\n", pd.DataFrame(final_result))

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, fbeta_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
MLFLOW_CLIENT = MlflowClient()

FEATURE_COLUMNS = {
    "full_features": [
        "credit_lines_outstanding",
        "loan_amt_outstanding",
        "total_debt_outstanding",
        "income",
        "years_employed",
        "fico_score",
    ],
    "safe_features": [
        "loan_amt_outstanding",
        "income",
        "years_employed",
        "fico_score",
    ],
}


def choose_dataset_mode():
    return ["full_features", "safe_features"]


def load_test_split(split_dir):
    X_test = pd.read_csv(f"{split_dir}/X_test.csv")
    y_test = pd.read_csv(f"{split_dir}/y_test.csv").values.ravel()
    return X_test, y_test


training_experiments = {
    "Logistic_Regression": "Loan_Default_Logistic_Regression_Local",
    "Random_Forest": "Loan_Default_Random_Forest_Local",
    "XGBoost": "Loan_Default_XGBoost_Local",
}


def model_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "estimator") and hasattr(model.estimator, "feature_names_in_"):
        return list(model.estimator.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    if hasattr(model, "estimator") and hasattr(model.estimator, "named_steps"):
        for step in model.estimator.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


def get_model_artifact_path(run_id):
    artifact_uri = MLFLOW_CLIENT.get_run(run_id).info.artifact_uri
    if artifact_uri.startswith("file:"):
        artifact_uri = artifact_uri.removeprefix("file:")
    return str(Path(artifact_uri) / "model")


def get_latest_finished_run(experiment_name, dataset_mode):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    expected_columns = FEATURE_COLUMNS[dataset_mode]
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            f"and params.feature_mode = '{dataset_mode}'"
        ),
        order_by=["metrics.val_f2 DESC"],
        max_results=20,
    )

    for _, run in runs.iterrows():
        try:
            model = mlflow.sklearn.load_model(get_model_artifact_path(run["run_id"]))
        except MlflowException:
            continue

        feature_names = model_feature_names(model)
        if feature_names is not None and feature_names != expected_columns:
            continue

        return {
            "run_id": run["run_id"],
            "experiment_id": experiment.experiment_id,
            "threshold": float(run["params.optimized_threshold"]),
            "val_f2": float(run["metrics.val_f2"]),
        }

    raise ValueError(
        f"No valid FINISHED runs found for experiment: {experiment_name} "
        f"with feature mode {dataset_mode}"
    )


final_result = []
dataset_modes = choose_dataset_mode()

for dataset_mode in dataset_modes:
    split_dir = f"data/processed/{dataset_mode}"
    X_test, y_test = load_test_split(split_dir)

    print(f"\nEvaluation dataset mode: {dataset_mode}")
    print("starting final evaluation for the three models")

    for name, experiment_name in training_experiments.items():
        info = get_latest_finished_run(experiment_name, dataset_mode)

        print(
            f"Selected model ({name}) - Run ID: {info['run_id']} "
            f"| Val F2: {info['val_f2']:.4f}"
        )

        with mlflow.start_run(
            run_id=info["run_id"],
            nested=True,
            run_name=f"Test_Eval_{name}_{dataset_mode}",
        ):
            mlflow.log_param("test_evaluation_mode", dataset_mode)
            model = mlflow.sklearn.load_model(get_model_artifact_path(info["run_id"]))

            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= info["threshold"]).astype(int)

            metrics = {
                "test_f1": f1_score(y_test, y_pred),
                "test_f2": fbeta_score(y_test, y_pred, beta=2),
                "test_recall": recall_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_accuracy": accuracy_score(y_test, y_pred),
            }

            mlflow.log_metrics(metrics)

            final_result.append({"Model": name, "Mode": dataset_mode, **metrics})
            print(f"{name} test metrics logged to MLflow (nested in {info['run_id']})")

print("\n", pd.DataFrame(final_result))

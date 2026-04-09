import json
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pathlib import Path

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, fbeta_score, classification_report, ConfusionMatrixDisplay

mlflow.set_tracking_uri("sqlite:///mlflow.db")
MLFLOW_CLIENT = MlflowClient()
ARTIFACT_TMP_DIR = Path("artifacts_tmp")
ARTIFACT_TMP_DIR.mkdir(exist_ok=True)
EVALUATION_EXPERIMENT = "Loan_Default_Evaluation_Local"

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


def tmp_artifact_path(filename):
    return ARTIFACT_TMP_DIR / filename


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

mlflow.set_experiment(EVALUATION_EXPERIMENT)

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
            run_name=f"Test_Eval_{name}_{dataset_mode}",
        ):
            mlflow.set_tags(
                {
                    "run_type": "evaluation",
                    "source_training_run_id": info["run_id"],
                    "source_training_experiment_id": info["experiment_id"],
                    "model_name": name,
                    "dataset_mode": dataset_mode,
                }
            )
            mlflow.log_param("source_training_run_id", info["run_id"])
            mlflow.log_param("source_training_experiment_id", info["experiment_id"])
            mlflow.log_param("test_evaluation_mode", dataset_mode)
            mlflow.log_param("model_name", name)
            mlflow.log_param("optimized_threshold", round(info["threshold"], 3))
            mlflow.log_metric("source_val_f2", info["val_f2"])
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

            # 1. Generate Confusion Matrix Plot
   

            mlflow.log_metrics(metrics)
            mlflow.log_metrics(metrics)

            # 2. Log Visual Artifacts (Confusion Matrix)
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
            plt.title(f"Confusion Matrix: {name} ({dataset_mode})")
            
            plot_path = tmp_artifact_path(f"cm_{name}_{dataset_mode}.png")
            plt.savefig(plot_path)
            plt.close(fig) 
            mlflow.log_artifact(str(plot_path))
            plot_path.unlink()

            # 3. Save the Classification Report (Detailed Precision/Recall per class)
            report_path = tmp_artifact_path(f"report_{name}_{dataset_mode}.txt")
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, y_pred))
            mlflow.log_artifact(str(report_path))
            report_path.unlink()

            # 4. Save the Metrics as a JSON (Great for automated reports)
            metrics_path = tmp_artifact_path(f"metrics_{name}_{dataset_mode}.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            mlflow.log_artifact(str(metrics_path))
            metrics_path.unlink()

            eval_metrics_path = tmp_artifact_path(f"eval_metrics_{name}_{dataset_mode}.json")
            metrics_payload = {
                "model_name": name,
                "dataset_mode": dataset_mode,
                "source_training_run_id": info["run_id"],
                "source_training_experiment_id": info["experiment_id"],
                "optimized_threshold": round(info["threshold"], 3),
                "source_val_f2": info["val_f2"],
                "test_metrics": metrics,
            }
            eval_metrics_path.write_text(
                json.dumps(metrics_payload, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(eval_metrics_path), artifact_path="evaluation")
            eval_metrics_path.unlink()

            predictions_path = tmp_artifact_path(f"test_predictions_{name}_{dataset_mode}.csv")
            predictions_df = X_test.copy()
            predictions_df["y_true"] = y_test
            predictions_df["y_proba"] = y_proba
            predictions_df["y_pred"] = y_pred
            predictions_df.to_csv(predictions_path, index=False)
            mlflow.log_artifact(str(predictions_path), artifact_path="evaluation")
            predictions_path.unlink()

            final_result.append({"Model": name, "Mode": dataset_mode, **metrics})
            print(
                f"{name} test metrics and artifacts logged to MLflow "
                f"(source training run: {info['run_id']})"
            )

print("\n", pd.DataFrame(final_result))

# Convert the list of dicts to a DataFrame
summary_df = pd.DataFrame(final_result)
summary_path = tmp_artifact_path("final_model_comparison.csv")
summary_df.to_csv(summary_path, index=False)
mlflow.log_artifact(str(summary_path), artifact_path="evaluation")
summary_path.unlink()

# Log the master table to a "Global" run or just keep it locally for your report
print("\n--- FINAL SUMMARY TABLE ---")
print(summary_df.to_string(index=False))

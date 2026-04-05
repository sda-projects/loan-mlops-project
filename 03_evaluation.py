import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, fbeta_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Final_Test_Evaluation")

def choose_dataset_mode():
    return ["full_features", "safe_features"]


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

    # Recherche des runs terminés avec le bon mode, triés par score F2 décroissant
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            f"and params.feature_mode = '{dataset_mode}'"
        ),
        order_by=["metrics.val_f2 DESC"],
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
        "threshold": float(run["params.optimized_threshold"]),
        "val_f2": float(run["metrics.val_f2"])
    }


final_result = []
dataset_modes = choose_dataset_mode()

for dataset_mode in dataset_modes:
    split_dir = f"data/processed/{dataset_mode}"
    X_test, y_test = load_test_split(split_dir)

    print(f"\nEvaluation dataset mode: {dataset_mode}")
    print("starting final evaluation for the three models")

    for name, experiment_name in training_experiments.items():
        info = get_latest_finished_run(experiment_name, dataset_mode)
        
        # Affichage du Run ID et du score sélectionné
        print(f"Modèle sélectionné ({name}) - Run ID : {info['run_id']} | Val F2: {info['val_f2']:.4f}")

        with mlflow.start_run(run_name=f"Test_Eval_{name}_{dataset_mode}"):
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

            final_result.append({"Model": name, "Mode": dataset_mode, **metrics})
            print(f"{name} Test metrics logged to MLflow!")

print("\n", pd.DataFrame(final_result))

import pandas as pd
import os 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# Signification des colonnes brutes du dataset :
# - credit_lines_outstanding : Nombre total de lignes de crédit ouvertes par le client (ex: cartes de crédit, prêts personnels, lignes de découvert).
# - loan_amt_outstanding : Montant principal encore dû sur le prêt actuel qui fait l'objet de l'évaluation de risque.
# - total_debt_outstanding : Dette totale cumulée du client (toutes dettes confondues : crédits, prêts immobiliers, emprunts divers).
# - income : Revenu annuel brut du client.
# - years_employed : Ancienneté professionnelle exprimée en nombre d'années dans l'emploi actuel.
# - fico_score : Score de crédit FICO, outil standard aux USA pour évaluer la probabilité de remboursement (valeurs généralement entre 300 et 850). Plus ce score est élevé, plus le client est considéré comme solvable.
# - default : Variable cible binaire (1 si le client est en défaut de paiement, 0 s'il rembourse normalement).


def load_clean_data(filepath):
    # Load the dataset.
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Basic cleaning.
    df = df.dropna()
    df = df.drop_duplicates()

    return df 


def select_feature_columns(feature_df, target_column, feature_mode):
    # En mode 'full', on prend toutes les colonnes originales.
    # En mode 'safe', on restreint à une sous-sélection (ex: sans total_debt_outstanding).
    full_columns = [
        "credit_lines_outstanding",
        "loan_amt_outstanding",
        "total_debt_outstanding",
        "income",
        "years_employed",
        "fico_score",
    ]
    safe_columns = [
        "loan_amt_outstanding",
        "income",
        "years_employed",
        "fico_score",
    ]

    feature_sets = {
        "full_features": full_columns,
        "safe_features": safe_columns,
    }
    
    if feature_mode not in feature_sets:
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    X = feature_df[feature_sets[feature_mode]].copy()
    y = feature_df[target_column]
    return X, y


def split_data(df, target_column, feature_mode):
    # Keep the target out of the feature matrix.
    X, y = select_feature_columns(df, target_column, feature_mode)

    # First split: pull out the test set (15%).
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Second split: train (70% total) and validation (15% total).
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    # Don't scale here, let the Pipeline in training handle it.
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_val = pd.DataFrame(X_val, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_split(output_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)


# Apply the cleaning
if __name__ == "__main__":
    raw_data_path = "data/Loan_Data.csv"
    output_root = "data/processed"
    feature_modes = ["full_features", "safe_features"]

    print("Start preprocessing")
    df = load_clean_data(raw_data_path)

    for feature_mode in feature_modes:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df, "default", feature_mode
        )
        output_dir = os.path.join(output_root, feature_mode)
        save_processed_split(output_dir, X_train, X_val, X_test, y_train, y_val, y_test)
        print(f"Preprocessing complete. Files saved in {output_dir}")
        print(f"Feature mode: {feature_mode}")
        print(f"Selected feature count: {X_train.shape[1]}")


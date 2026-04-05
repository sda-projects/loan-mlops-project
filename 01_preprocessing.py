import pandas as pd
import os 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler


def load_clean_data(filepath):
    # Load the dataset.
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Basic cleaning.
    df = df.dropna()
    df = df.drop_duplicates()

    return df 


def add_engineered_features(df):
    feature_df = df.copy()
    eps = 1e-6

    # Ratios derived from the raw financial variables can add signal without changing the source data.
    feature_df["debt_to_income"] = feature_df["total_debt_outstanding"] / (feature_df["income"] + eps)
    feature_df["loan_to_income"] = feature_df["loan_amt_outstanding"] / (feature_df["income"] + eps)
    feature_df["loan_to_debt"] = feature_df["loan_amt_outstanding"] / (feature_df["total_debt_outstanding"] + eps)
    feature_df["debt_per_credit_line"] = feature_df["total_debt_outstanding"] / (
        feature_df["credit_lines_outstanding"] + eps
    )
    feature_df["income_per_credit_line"] = feature_df["income"] / (
        feature_df["credit_lines_outstanding"] + eps
    )
    feature_df["credit_lines_per_year"] = feature_df["credit_lines_outstanding"] / (
        feature_df["years_employed"] + 1
    )
    feature_df["loan_per_year_employed"] = feature_df["loan_amt_outstanding"] / (
        feature_df["years_employed"] + 1
    )
    feature_df["fico_income_interaction"] = feature_df["fico_score"] * feature_df["income"]
    feature_df["fico_debt_interaction"] = feature_df["fico_score"] * feature_df["debt_to_income"]

    return feature_df


def select_feature_columns(feature_df, target_column, feature_mode):
    full_feature_columns = [
        "credit_lines_outstanding",
        "loan_amt_outstanding",
        "total_debt_outstanding",
        "income",
        "years_employed",
        "fico_score",
        "debt_to_income",
        "loan_to_income",
        "loan_to_debt",
        "debt_per_credit_line",
        "income_per_credit_line",
        "credit_lines_per_year",
        "loan_per_year_employed",
        "fico_income_interaction",
        "fico_debt_interaction",
    ]
    safe_feature_columns = [
        "loan_amt_outstanding",
        "income",
        "years_employed",
        "fico_score",
        "loan_to_income",
        "loan_per_year_employed",
        "fico_income_interaction",
    ]

    feature_sets = {
        "full_features": full_feature_columns,
        "safe_features": safe_feature_columns,
    }
    if feature_mode not in feature_sets:
        raise ValueError(
            f"Unknown feature mode: {feature_mode}. "
            f"Expected one of {list(feature_sets)}."
        )

    selected_columns = feature_sets[feature_mode]
    X = feature_df[selected_columns].copy()
    y = feature_df[target_column]
    return X, y


def split_data(df, target_column, feature_mode):
    feature_df = add_engineered_features(df)

    # Keep the target out of the feature matrix and switch feature sets explicitly.
    X, y = select_feature_columns(feature_df, target_column, feature_mode)

    # First split: pull out the test set (15%).
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Second split: train (70% total) and validation (15% total).
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    # Fit scaling only on the training fold, then reuse it for validation and test.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve feature names.
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


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


import pandas as pd
import os 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

def load_clean_data(filepath):
    #load the dataset 
    df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip()

    #basic cleaning: drop duplicates, handle missing values (adapt - no nans in this database)

    df = df.dropna()
    df = df.drop_duplicates()

    return df 

def split_data(df, target_column):

    
    # 2. Define Features and Target (Dropping Leaks)
    to_drop = [target_column, 'customer_id', 'credit_lines_outstanding', 'total_debt_outstanding']
    X = df.drop(columns=to_drop, errors='ignore')
    y = df[target_column]

    # 3. First split: Pull out the Test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # 4. Second split: Split temp into Train (70% total) and Val (15% total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    # 5. Scaling (The "Golden Rule": Fit only on Train)
    scaler = StandardScaler()
    
    # Fit and transform the Training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # ONLY transform the Validation and Test data (no fitting!)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Optional: Convert back to DataFrame to keep column names for MLflow
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


# Apply the cleaning 
if __name__ == "__main__":
    #define paths 
    raw_data_path = "data/Loan_Data.csv"
    output_dir = 'data/processed'

    #run the functions 
    print("Start preprocessing")
    df = load_clean_data(raw_data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, "default")

    #save the results 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
   

    print(f"Preprocesssing complete. Files saved in {output_dir}")


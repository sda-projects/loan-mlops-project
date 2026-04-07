import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from preprocessing import load_clean_data, split_data

def test_load_clean_data():
    df = load_clean_data("data/Loan_Data.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_split_data():
    df = load_clean_data("data/Loan_Data.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, "default", "full_features")

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    assert len(y_train) > 0
    assert len(y_val) > 0
    assert len(y_test) > 0

    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
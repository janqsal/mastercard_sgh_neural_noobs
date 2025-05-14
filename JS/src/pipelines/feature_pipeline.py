import pandas as pd
from src.features.prepare import prepare_model_data


def run_feature_engineering(
    processed_data_path: str,
    X_train_path: str,
    X_test_path: str,
    y_train_path: str,
    y_test_path: str,
    **prepare_kwargs,
):
    """Load processed data, generate feature matrices, and save them to Parquet files."""
    df = pd.read_parquet(processed_data_path)

    X_train, y_train, X_test, y_test = prepare_model_data(df, **prepare_kwargs)

    X_train.to_parquet(X_train_path, index=False)
    X_test.to_parquet(X_test_path, index=False)
    y_train.to_frame("y").to_parquet(y_train_path, index=False)
    y_test.to_frame("y").to_parquet(y_test_path, index=False)

    return X_train, y_train, X_test, y_test

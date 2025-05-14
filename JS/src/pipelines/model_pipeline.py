from pathlib import Path
import pandas as pd
from src.modeling import train_model, save_model, auc_score


def run_model_pipeline(
    X_train_path: str,
    y_train_path: str,
    X_test_path: str,
    y_test_path: str,
    model_output_path: str,
    *,
    oversample: bool = True,
    xgb_params: dict | None = None,
):
    """Train model, evaluate AUC, and return all objects needed for downstream notebook validation."""

    # load datasets
    X_train = pd.read_parquet(X_train_path)
    y_train = pd.read_parquet(y_train_path)["y"]
    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path)["y"]

    # train
    model, X_res, y_res = train_model(
        X_train,
        y_train,
        oversample_enabled=oversample,
        xgb_params=xgb_params,
    )

    # metrics
    auc_train = auc_score(model, X_res, y_res)
    auc_test = auc_score(model, X_test, y_test)

    # save
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, model_output_path)

    return {
        "model": model,
        "auc_train": auc_train,
        "auc_test": auc_test,
        "X_train_oversampled": X_res,
        "y_train_oversampled": y_res,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

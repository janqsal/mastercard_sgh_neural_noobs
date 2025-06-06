import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from .build import build_model
from .oversample import oversample

__all__ = [
    "train_model",
    "save_model",
    "load_model",
    "predict_proba",
    "auc_score",
    "accuracy_cls",
]


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    oversample_enabled: bool = True,
    xgb_params: dict | None = None,
):
    """Train an XGBoost model; optional RandomOverSampler."""
    if oversample_enabled:
        X_res, y_res = oversample(X_train, y_train)
    else:
        X_res, y_res = X_train, y_train

    model = build_model(xgb_params)
    model.fit(X_res, y_res)
    return model, X_res, y_res


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


def predict_proba(model, X: pd.DataFrame):
    return model.predict_proba(X)[:, 1]


def auc_score(model, X: pd.DataFrame, y_true: pd.Series) -> float:
    return roc_auc_score(y_true, predict_proba(model, X))


def accuracy_cls(model, X: pd.DataFrame, y_true: pd.Series) -> float:
    return accuracy_score(y_true, model.predict(X))

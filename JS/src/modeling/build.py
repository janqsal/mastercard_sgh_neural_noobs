from xgboost import XGBClassifier

DEFAULT_XGB_PARAMS = {
    "n_estimators": 1250,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.5,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "eval_metric": "auc",
    "random_state": 42,
}


def build_model(params: dict | None = None) -> XGBClassifier:
    """Create and return an XGBClassifier with provided parameters or defaults."""
    cfg = DEFAULT_XGB_PARAMS.copy()
    if params:
        cfg.update(params)
    return XGBClassifier(**cfg)

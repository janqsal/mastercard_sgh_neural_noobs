# tune.py  ── minimal Optuna study around your run_model_pipeline
import optuna
from pathlib import Path
from src.pipelines.model_pipeline import run_model_pipeline

DATA_DIR   = "../data"
MODELS_DIR = "../models/optuna"
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

def objective(trial: optuna.Trial) -> float:
    """Return test-set AUC for a sampled XGBoost configuration."""
    params = {
        "n_estimators"     : trial.suggest_int ("n_estimators", 100, 1000, step=20),
        "learning_rate"    : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth"        : trial.suggest_int ("max_depth", 2, 6),
        "subsample"        : trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha",  1e-8, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "eval_metric"      : "auc",
        "random_state"     : 42,
    }

    # every trial writes its model (optional – you can skip saving if disk I/O hurts)
    model_path = MODELS_DIR + f"/xgb_trial_{trial.number}.joblib"

    results = run_model_pipeline(
        X_train_path=f"{DATA_DIR}/X_train.parquet",
        y_train_path=f"{DATA_DIR}/y_train.parquet",
        X_test_path =f"{DATA_DIR}/X_test.parquet",
        y_test_path =f"{DATA_DIR}/y_test.parquet",
        model_output_path=model_path,
        oversample=True,
        xgb_params=params,
    )
    return results["auc_test"]          # Optuna will maximise this

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=None)   # run all night

    print("Best AUC :", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    study.trials_dataframe().to_csv(MODELS_DIR + "/optuna_trials.csv", index=False)

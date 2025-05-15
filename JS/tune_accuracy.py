import optuna
from pathlib import Path
from src.pipelines.model_pipeline import run_model_pipeline

DATA_DIR   = "../data"
MODELS_DIR = "../models/optuna_acc3"
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators"     : trial.suggest_int ("n_estimators", 1800, 2500, step=25),
        "learning_rate"    : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth"        : trial.suggest_int ("max_depth", 9, 15),
        "subsample"        : trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "eval_metric"      : "logloss",
        "random_state"     : 42,
    }

    model_path = f"{MODELS_DIR}/xgb_trial_{trial.number}.joblib"

    results = run_model_pipeline(
        X_train_path=f"{DATA_DIR}/X_train.parquet",
        y_train_path=f"{DATA_DIR}/y_train.parquet",
        X_test_path =f"{DATA_DIR}/X_test.parquet",
        y_test_path =f"{DATA_DIR}/y_test.parquet",
        model_output_path=model_path,
        oversample=True,
        xgb_params=params,
    )

    return results["accuracy_test"]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=70)

    print("Best accuracy:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    study.trials_dataframe().to_csv(f"{MODELS_DIR}/trials.csv", index=False)

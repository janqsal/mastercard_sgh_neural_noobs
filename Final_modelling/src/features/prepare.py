import pandas as pd
from datetime import datetime


def prepare_model_data(
    df: pd.DataFrame,
    *,
    to_drop: list,
    to_think_but_drop: list,
    to_categorize: list,
    cutoff: str = "2023-07-01",
    target: str = "is_fraud",
):
    """

    Steps:
    1. Remove the first transaction of each user.
    2. Drop columns listed in *to_drop* and *to_think_but_drop*.
    3. One‑hot encode columns listed in *to_categorize* (no drop_first).
    4. Cast boolean columns to int.
    5. Split the data by *cutoff* timestamp (< for train, >= for test).
    6. Compute per‑merchant fraud rate (bad_rate) and merge into X.
    7. Drop helper columns (merchant_id, total_transactions, num_frauds, timestamp).

    Returns
    -------
    X_train, y_train, X_test, y_test : tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
    """
    df = df.copy()

    # Step 1: remove the earliest transaction per user
    first_idx = (
        df.sort_values(["user_id", "timestamp"])  # sort chronologically per user
        .groupby("user_id")
        .head(1)
        .index
    )
    df = df.drop(index=first_idx)

    # Step 2: drop columns
    df = df.drop(columns=to_drop, errors="ignore")
    df = df.drop(columns=to_think_but_drop, errors="ignore")

    # Step 3: one-hot encode categoricals
    df = pd.get_dummies(df, columns=to_categorize, drop_first=False)

    # Step 4: boolean → int
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)

    # Step 5: split by timestamp
    cutoff_ts = pd.to_datetime(cutoff)
    df_train = df[df["timestamp"] < cutoff_ts]
    df_test = df[df["timestamp"] >= cutoff_ts]

    # Step 6: per‑merchant bad rate
    bad = (
        df.groupby("merchant_id")[target]
        .agg(total_transactions="count", num_frauds="sum")
    )
    bad["bad_rate"] = bad["num_frauds"] / bad["total_transactions"]

    def _add_bad_rate(X: pd.DataFrame) -> pd.DataFrame:
        X = X.merge(bad, on="merchant_id", how="left")
        return X.drop(["merchant_id", "total_transactions", "num_frauds"], axis=1)

    X_train = _add_bad_rate(df_train).drop(columns=[target, "timestamp"])
    X_test = _add_bad_rate(df_test).drop(columns=[target, "timestamp"])

    y_train = df_train[target]
    y_test = df_test[target]

    return X_train, y_train, X_test, y_test

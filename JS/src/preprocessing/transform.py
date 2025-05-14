import pandas as pd
from src.preprocessing.helpers import get_part_of_day, haversine, flag_within_percent


def data_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply notebook preprocessing steps to a DataFrame (mutates a copy)."""
    df = df.copy()

    # ----- Timestamp‑derived columns -----
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["Month_Year_EOM"] = (df["timestamp"] + pd.offsets.MonthEnd(0)).dt.date
    df["Date"] = df["timestamp"].dt.strftime("%d-%m-%Y")
    df["Year"] = df["timestamp"].dt.strftime("%Y")
    df["hour"] = df["timestamp"].dt.hour
    df["part_of_day"] = df["hour"].apply(get_part_of_day)

    # Ensure user‑ordered history
    df = df.sort_values(["user_id", "timestamp"])

    # ----- Time‑difference features -----
    df["time_diff"] = df.groupby("user_id")["timestamp"].diff()
    df["time_diff_hours"] = (df["time_diff"].dt.total_seconds() / 3600).round(2)

    # ----- Lat / Lon extraction from nested dict -----
    df["latitude"] = df["location"].apply(lambda x: x.get("lat")).round(2)
    df["longitude"] = df["location"].apply(lambda x: x.get("long")).round(2)

    # Shift lat / lon / timestamp per user to get previous point
    df["lat_prev"] = df.groupby("user_id")["latitude"].shift()
    df["lon_prev"] = df.groupby("user_id")["longitude"].shift()
    df["time_prev"] = df.groupby("user_id")["timestamp"].shift()

    # Distance and speed
    df["distance_km"] = haversine(df["lat_prev"], df["lon_prev"], df["latitude"], df["longitude"])
    df["speed_kmph"] = df["distance_km"] / df["time_diff_hours"]
    df["speed_kmph"] = df["speed_kmph"].clip(upper=2000)

    # Amount ratios
    df["amount/avg_amount"] = df["amount"] / df["avg_transaction_amount"]
    df["amount/sum_monthly_installments"] = df["amount"] / df["sum_of_monthly_installments"]
    df["amount/sum_monthly_expenses"] = df["amount"] / df["sum_of_monthly_expenses"]

    # Country comparison flags
    df["country_u=t"] = (df["country_users"] == df["transaction_country"]).astype(bool)
    df["country_m=t"] = (df["country_merchant"] == df["transaction_country"]).astype(bool)
    df["countries_same"] = (df["country_merchant"] == df["country_users"]).astype(int)

    # ±10 % and ±5 % amount consistency flags
    df = flag_within_percent(df, amount_col="amount", user_col="user_id", tolerance_pct=10)
    df = flag_within_percent(df, amount_col="amount", user_col="user_id", tolerance_pct=5)

    # Remove the very first row per user (as w notebook – no previous history)
    first_idx = df.sort_values(["user_id", "timestamp"]).groupby("user_id").head(1).index
    df = df.drop(index=first_idx)

    return df

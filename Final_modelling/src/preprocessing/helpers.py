import numpy as np
import pandas as pd

# -------------------------------
# 1. Part‑of‑day categorisation
# -------------------------------

def get_part_of_day(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    elif 21 <= hour < 23:
        return "night"
    else:
        return "late_night"

# -------------------------------
# 2. Haversine distance (km)
# -------------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# -------------------------------
# 3. Flag within ±X % of previous amount
# -------------------------------

def flag_within_percent(df: pd.DataFrame, amount_col: str = "amount", user_col: str = "user_id", tolerance_pct: int = 10) -> pd.DataFrame:
    """Replicates the exact function from the notebook."""
    df = df.copy()
    flag_col = f"within_{tolerance_pct}pct"

    def _per_user(group: pd.DataFrame):
        previous = group[amount_col].shift(1)
        lower = previous * (1 - tolerance_pct / 100)
        upper = previous * (1 + tolerance_pct / 100)
        flag = ((group[amount_col] >= lower) & (group[amount_col] <= upper)).astype(float)
        flag.iloc[0] = np.nan  # first tx per user has no previous reference
        return flag

    df = df.sort_values([user_col, "timestamp"])
    df[flag_col] = df.groupby(user_col, group_keys=False).apply(_per_user)
    return df

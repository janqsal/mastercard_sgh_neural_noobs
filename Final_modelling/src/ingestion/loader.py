import pandas as pd


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load and merge merchants, users, transactions, and geo data into a single DataFrame.

    :param data_dir: Path to the directory containing data files.
    :return: Merged DataFrame with all relevant fields.
    """
    # Load datasets
    merchants = pd.read_csv(f"{data_dir}/merchants.csv")
    users = pd.read_csv(f"{data_dir}/users.csv").rename(columns={"country": "country_users"})
    merchants = merchants.rename(columns={"country": "country_merchant"})
    transactions = pd.read_json(
        f"{data_dir}/transactions.json", lines=True, dtype_backend="numpy_nullable"
    )
    geo_df = pd.read_csv(f"{data_dir}/geo_df.csv")

    # Merge datasets step by step
    df = transactions.merge(merchants, on="merchant_id", how="left")
    df = df.merge(users, on="user_id", how="left")
    df = df.merge(geo_df, on="transaction_id", how="left")

    return df

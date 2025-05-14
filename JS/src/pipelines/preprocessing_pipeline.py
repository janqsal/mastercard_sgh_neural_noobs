import pandas as pd
from src.preprocessing.transform import data_transform


def run_preprocessing(raw_data_path: str, output_path: str) -> pd.DataFrame:
    """
    1. Load merged raw data (Parquet) produced by ingestion.
    2. Apply `data_transform` (identical to notebook logic).
    3. Save the processed DataFrame to Parquet.

    :param raw_data_path: Path to Parquet file with merged raw data.
    :param output_path: Path where the processed data will be saved.
    :return: Processed DataFrame.
    """
    df = pd.read_parquet(raw_data_path)
    processed_df = data_transform(df)
    processed_df.to_parquet(output_path, index=False)
    return processed_df
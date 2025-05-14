import pandas as pd
from src.ingestion.loader import load_data


def run_ingestion(data_dir: str, output_path: str) -> pd.DataFrame:
    """
    Pipeline step: load raw data and save as Parquet for downstream processing.

    :param data_dir: Path to raw data directory containing CSV/JSON files.
    :param output_path: Path where the merged raw DataFrame will be saved (Parquet format).
    :return: Loaded DataFrame.
    """
    df = load_data(data_dir)
    df.to_parquet(output_path, index=False)
    return df
